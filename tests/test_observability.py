"""Tests for optional Phoenix tracing helpers."""

import argparse
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from nemori.config import MemoryConfig
from nemori.observability import PhoenixTracing, _TracingSettings, setup_phoenix_tracing
from scripts.verify_phoenix_traces import _fetch_spans


def test_setup_is_noop_when_tracing_is_disabled():
    assert setup_phoenix_tracing(MemoryConfig(enable_llm_tracing=False)) is None


def test_tenant_context_contains_both_tenant_keys():
    using_attributes = MagicMock(return_value=nullcontext())
    runtime = PhoenixTracing(
        provider=MagicMock(),
        tracer=MagicMock(),
        using_attributes=using_attributes,
        settings=_TracingSettings("http://localhost", "nemori", None),
    )

    runtime.tenant_context("user-a", "agent-a")

    using_attributes.assert_called_once_with(
        user_id="agent-a:user-a",
        metadata={"agent_id": "agent-a", "user_id": "user-a"},
        tags=["nemori", "memory"],
    )


def test_force_flush_is_non_fatal():
    provider = MagicMock()
    provider.force_flush.side_effect = RuntimeError("collector down")
    runtime = PhoenixTracing(
        provider=provider,
        tracer=MagicMock(),
        using_attributes=MagicMock(),
        settings=_TracingSettings("http://localhost", "nemori", None),
    )

    assert runtime.force_flush() is False


def test_tracing_settings_include_privacy_controls_and_hide_api_key():
    settings = _TracingSettings(
        endpoint="https://collector.example.com/v1/traces",
        project_name="nemori",
        api_key="secret-key",
        hide_inputs=False,
        hide_outputs=True,
        hide_embeddings_vectors=False,
        hide_embeddings_text=True,
    )

    assert settings.hide_inputs is False
    assert settings.hide_outputs is True
    assert "secret-key" not in repr(settings)


@pytest.mark.parametrize(
    "base_url",
    [
        "file:///tmp/traces.json",
        "http://collector.example.com",
        "https://user:password@collector.example.com",
    ],
)
def test_trace_verifier_rejects_unsafe_base_urls(base_url):
    args = argparse.Namespace(
        base_url=base_url,
        project="nemori",
        start_time="2026-07-18T00:00:00Z",
    )

    with pytest.raises(ValueError):
        _fetch_spans(args)
