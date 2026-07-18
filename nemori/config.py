# src/config.py
"""Nemori configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from ipaddress import ip_address
from urllib.parse import urlparse

from nemori.domain.exceptions import ConfigError


def _resolve_llm_key() -> str:
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )


def _resolve_embedding_key() -> str:
    return os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def _resolve_embedding_headers() -> dict[str, str] | None:
    modal_key = os.getenv("MODAL_PROXY_AUTH_TOKEN_ID")
    modal_secret = os.getenv("MODAL_PROXY_AUTH_TOKEN_SECRET")
    base_url = os.getenv("EMBEDDING_BASE_URL") or ""
    parsed = urlparse(base_url)
    hostname = parsed.hostname or ""
    if (
        modal_key
        and modal_secret
        and parsed.scheme.lower() == "https"
        and hostname.endswith(".modal.run")
    ):
        return {"Modal-Key": modal_key, "Modal-Secret": modal_secret}
    return None


def _resolve_dsn() -> str:
    return (
        os.getenv("DATABASE_URL")
        or os.getenv("DSN")
        or "postgresql://nemori:nemori@localhost:5432/nemori"
    )


def _resolve_base_url(env_key: str) -> str | None:
    return os.getenv(env_key) or None


def _resolve_bool(env_key: str, default: bool = False) -> bool:
    value = os.getenv(env_key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{env_key} must be a boolean value")


def _is_local_hostname(hostname: str) -> bool:
    """Return whether an HTTP endpoint is limited to a local/private network."""
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or "." not in normalized:
        return True
    try:
        address = ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def _validate_http_endpoint(
    name: str,
    endpoint: str | None,
    *,
    require_https_for_public: bool = False,
) -> None:
    """Reject malformed endpoints and optionally require secure public transport."""
    if endpoint is None:
        return
    if not isinstance(endpoint, str):
        raise ConfigError(f"{name} must be a URL")

    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"{name} must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(f"{name} must not contain credentials")
    if (
        require_https_for_public
        and parsed.scheme == "http"
        and not _is_local_hostname(parsed.hostname)
    ):
        raise ConfigError(f"{name} must use HTTPS for non-local hosts")


@dataclass
class MemoryConfig:
    """Configuration for the Nemori memory system."""

    # Database
    dsn: str = field(default_factory=_resolve_dsn, repr=False)
    db_pool_min: int = 5
    db_pool_max: int = 20

    # Multi-tenant isolation
    agent_id: str = "default"

    # LLM
    llm_model: str = "qwen3.6-27b-fp8"
    llm_api_key: str = field(default_factory=_resolve_llm_key, repr=False)
    llm_base_url: str | None = field(
        default_factory=lambda: _resolve_base_url("LLM_BASE_URL")
    )
    llm_max_concurrent: int = 10
    llm_timeout: float = 30.0
    llm_retries: int = 3
    llm_token_budget: int | None = None

    # Optional Phoenix/OpenInference tracing
    enable_llm_tracing: bool = field(
        default_factory=lambda: _resolve_bool("NEMORI_ENABLE_LLM_TRACING")
    )
    phoenix_collector_endpoint: str = field(
        default_factory=lambda: os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        or "http://localhost:6006/v1/traces"
    )
    phoenix_project_name: str = field(
        default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME") or "nemori"
    )
    phoenix_api_key: str | None = field(
        default_factory=lambda: os.getenv("PHOENIX_API_KEY"),
        repr=False,
    )
    trace_hide_inputs: bool = field(
        default_factory=lambda: _resolve_bool("OPENINFERENCE_HIDE_INPUTS", True)
    )
    trace_hide_outputs: bool = field(
        default_factory=lambda: _resolve_bool("OPENINFERENCE_HIDE_OUTPUTS", True)
    )
    trace_hide_embeddings_vectors: bool = field(
        default_factory=lambda: _resolve_bool(
            "OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS", True
        )
    )
    trace_hide_embeddings_text: bool = field(
        default_factory=lambda: _resolve_bool(
            "OPENINFERENCE_HIDE_EMBEDDINGS_TEXT", True
        )
    )

    # Embedding
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
    )
    embedding_api_key: str = field(default_factory=_resolve_embedding_key, repr=False)
    embedding_base_url: str | None = field(
        default_factory=lambda: _resolve_base_url("EMBEDDING_BASE_URL")
    )
    embedding_headers: dict[str, str] | None = field(
        default_factory=_resolve_embedding_headers,
        repr=False,
    )
    embedding_dimension: int = 1536
    embedding_warmup_timeout: float = 60.0
    embedding_warmup_retries: int = 2

    # Qdrant
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL") or "localhost",
        repr=False,
    )
    qdrant_port: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333"))
    )
    qdrant_api_key: str | None = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY"),
        repr=False,
    )
    qdrant_collection_prefix: str = "nemori"

    # Buffer & Generation
    buffer_size_min: int = 2
    buffer_size_max: int = 25
    enable_batch_segmentation: bool = True
    batch_threshold: int = 20
    episode_min_messages: int = 2
    episode_max_messages: int = 25

    # Semantic Memory
    enable_semantic_memory: bool = True
    enable_prediction_correction: bool = True
    semantic_similarity_threshold: float = 0.85

    # Episode Merging
    enable_episode_merging: bool = True
    merge_similarity_threshold: float = 0.85
    merge_top_k: int = 5

    # Search
    search_top_k_episodes: int = 10
    search_top_k_semantic: int = 10

    def __post_init__(self) -> None:
        if self.db_pool_min < 1:
            raise ConfigError("db_pool_min must be >= 1")
        if self.db_pool_max < self.db_pool_min:
            raise ConfigError("db_pool_max must be >= db_pool_min")
        if self.buffer_size_min < 1:
            raise ConfigError("buffer_size_min must be >= 1")
        if self.buffer_size_max < self.buffer_size_min:
            raise ConfigError("buffer_size_max must be >= buffer_size_min")
        if self.search_top_k_episodes < 1:
            raise ConfigError("search_top_k_episodes must be >= 1")
        if self.search_top_k_semantic < 1:
            raise ConfigError("search_top_k_semantic must be >= 1")
        if self.embedding_warmup_timeout <= 0:
            raise ConfigError("embedding_warmup_timeout must be > 0")
        if self.embedding_warmup_retries < 0:
            raise ConfigError("embedding_warmup_retries must be >= 0")
        _validate_http_endpoint("llm_base_url", self.llm_base_url)
        _validate_http_endpoint("embedding_base_url", self.embedding_base_url)
        if self.embedding_headers and {
            key.casefold() for key in self.embedding_headers
        } & {"modal-key", "modal-secret"}:
            parsed_embedding_url = urlparse(self.embedding_base_url or "")
            embedding_hostname = parsed_embedding_url.hostname or ""
            if (
                parsed_embedding_url.scheme.lower() != "https"
                or not embedding_hostname.endswith(".modal.run")
            ):
                raise ConfigError(
                    "Modal auth headers require an HTTPS .modal.run embedding_base_url"
                )
        if self.enable_llm_tracing and not self.phoenix_collector_endpoint:
            raise ConfigError("phoenix_collector_endpoint must not be empty")
        if self.enable_llm_tracing and not self.phoenix_project_name:
            raise ConfigError("phoenix_project_name must not be empty")
        if self.enable_llm_tracing:
            _validate_http_endpoint(
                "phoenix_collector_endpoint",
                self.phoenix_collector_endpoint,
                require_https_for_public=True,
            )
