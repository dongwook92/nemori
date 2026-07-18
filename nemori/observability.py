"""Optional process-wide Phoenix/OpenInference tracing."""

from __future__ import annotations

import atexit
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from nemori.config import MemoryConfig
from nemori.domain.exceptions import ConfigError

logger = logging.getLogger("nemori")


@dataclass(frozen=True)
class _TracingSettings:
    endpoint: str
    project_name: str
    api_key: str | None = field(repr=False)
    hide_inputs: bool = True
    hide_outputs: bool = True
    hide_embeddings_vectors: bool = True
    hide_embeddings_text: bool = True


class PhoenixTracing:
    """Shared Phoenix tracer and context helpers."""

    def __init__(
        self,
        provider: Any,
        tracer: Any,
        using_attributes: Any,
        settings: _TracingSettings,
    ) -> None:
        self.provider = provider
        self.tracer = tracer
        self._using_attributes = using_attributes
        self.settings = settings

    def tenant_context(self, user_id: str, agent_id: str) -> Any:
        """Propagate tenant identity to all nested OpenInference spans."""
        return self._using_attributes(
            user_id=f"{agent_id}:{user_id}",
            metadata={"agent_id": agent_id, "user_id": user_id},
            tags=["nemori", "memory"],
        )

    def start_span(self, name: str, attributes: dict[str, Any]) -> Any:
        return self.tracer.start_as_current_span(
            name,
            attributes={
                "openinference.span.kind": "CHAIN",
                **attributes,
            },
        )

    def force_flush(self, timeout_millis: int = 10_000) -> bool:
        """Flush queued spans without shutting down the process-global provider."""
        try:
            return bool(self.provider.force_flush(timeout_millis=timeout_millis))
        except Exception as exc:
            logger.warning("Failed to flush Phoenix traces: %s", exc)
            return False


_runtime: PhoenixTracing | None = None
_runtime_lock = Lock()
_atexit_registered = False


def setup_phoenix_tracing(config: MemoryConfig) -> PhoenixTracing | None:
    """Initialize Phoenix tracing once per process when explicitly enabled."""
    if not config.enable_llm_tracing:
        return None

    settings = _TracingSettings(
        endpoint=config.phoenix_collector_endpoint,
        project_name=config.phoenix_project_name,
        api_key=config.phoenix_api_key,
        hide_inputs=config.trace_hide_inputs,
        hide_outputs=config.trace_hide_outputs,
        hide_embeddings_vectors=config.trace_hide_embeddings_vectors,
        hide_embeddings_text=config.trace_hide_embeddings_text,
    )

    global _runtime, _atexit_registered
    with _runtime_lock:
        if _runtime is not None:
            if _runtime.settings != settings:
                raise ConfigError(
                    "Phoenix tracing is already initialized with different settings"
                )
            return _runtime

        try:
            from openinference.instrumentation import TraceConfig
            from openinference.instrumentation.openai import OpenAIInstrumentor
            from phoenix.otel import register, using_attributes
        except ImportError as exc:
            raise ConfigError(
                "Phoenix tracing dependencies are missing; install nemori[tracing]"
            ) from exc

        register_kwargs: dict[str, Any] = {
            "endpoint": settings.endpoint,
            "protocol": "http/protobuf",
            "project_name": settings.project_name,
            "batch": True,
            "verbose": False,
        }
        if settings.api_key:
            register_kwargs["api_key"] = settings.api_key

        provider = register(**register_kwargs)
        OpenAIInstrumentor().instrument(
            tracer_provider=provider,
            config=TraceConfig(
                hide_inputs=settings.hide_inputs,
                hide_outputs=settings.hide_outputs,
                hide_embeddings_vectors=settings.hide_embeddings_vectors,
                hide_embeddings_text=settings.hide_embeddings_text,
            ),
        )
        _runtime = PhoenixTracing(
            provider=provider,
            tracer=provider.get_tracer("nemori"),
            using_attributes=using_attributes,
            settings=settings,
        )
        if not _atexit_registered:
            atexit.register(_flush_at_exit)
            _atexit_registered = True
        logger.info(
            "Phoenix tracing enabled (project=%s, endpoint_host=%s)",
            settings.project_name,
            urlparse(settings.endpoint).hostname,
        )
        return _runtime


def _flush_at_exit() -> None:
    runtime = _runtime
    if runtime is not None:
        runtime.force_flush()
