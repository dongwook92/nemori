"""Tests for LLMOrchestrator."""

import pytest
import asyncio
from unittest.mock import AsyncMock
from nemori.llm.orchestrator import LLMOrchestrator, LLMRequest, LLMResponse, TokenUsage
from nemori.domain.exceptions import LLMError, TokenBudgetExceeded


class _RecordingSpan:
    def __init__(self):
        self.attributes = {}
        self.events = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes or {}))


class _SpanContext:
    def __init__(self, span):
        self.span = span

    def __enter__(self):
        return self.span

    def __exit__(self, exc_type, exc, traceback):
        return False


class _RecordingTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name, attributes):
        span = _RecordingSpan()
        self.spans.append((name, attributes, span))
        return _SpanContext(span)


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value="response text")
    provider.supports_usage_tracking = False
    return provider


@pytest.fixture
def orchestrator(mock_provider):
    return LLMOrchestrator(
        provider=mock_provider,
        default_model="gpt-4o-mini",
        max_concurrent=5,
    )


@pytest.mark.asyncio
async def test_execute_simple_request(orchestrator, mock_provider):
    request = LLMRequest(messages=({"role": "user", "content": "hi"},))
    response = await orchestrator.execute(request)
    assert isinstance(response, LLMResponse)
    assert response.content == "response text"
    mock_provider.complete.assert_called_once()


@pytest.mark.asyncio
async def test_execute_retries_on_error(mock_provider):
    call_count = 0

    async def flaky_complete(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise LLMError("server error")
        return "success"

    mock_provider.complete = flaky_complete
    mock_provider.supports_usage_tracking = False
    orch = LLMOrchestrator(provider=mock_provider, default_model="gpt-4o-mini")
    request = LLMRequest(messages=({"role": "user", "content": "hi"},), retries=3)
    response = await orch.execute(request)
    assert response.content == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_concurrency_limit():
    active = 0
    max_active = 0

    async def slow_complete(messages, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return "done"

    provider = AsyncMock()
    provider.complete = slow_complete
    provider.supports_usage_tracking = False
    orch = LLMOrchestrator(provider=provider, default_model="m", max_concurrent=2)
    requests = [
        LLMRequest(messages=({"role": "user", "content": f"{i}"},)) for i in range(5)
    ]
    await orch.execute_batch(requests)
    assert max_active <= 2


@pytest.mark.asyncio
async def test_token_budget_exceeded(mock_provider):
    mock_provider.supports_usage_tracking = False
    orch = LLMOrchestrator(provider=mock_provider, default_model="m", token_budget=100)
    orch._total_tokens = 100
    request = LLMRequest(messages=({"role": "user", "content": "hi"},))
    with pytest.raises(TokenBudgetExceeded):
        await orch.execute(request)


@pytest.mark.asyncio
async def test_stats_tracking(orchestrator, mock_provider):
    request = LLMRequest(messages=({"role": "user", "content": "hi"},))
    await orchestrator.execute(request)
    stats = orchestrator.stats
    assert stats.total_requests >= 1


@pytest.mark.asyncio
async def test_usage_tracking_with_provider():
    """Test that token usage is properly tracked when provider supports it."""
    provider = AsyncMock()
    provider.supports_usage_tracking = True
    provider.complete_with_usage = AsyncMock(
        return_value=("response text", {"prompt_tokens": 50, "completion_tokens": 30})
    )
    orch = LLMOrchestrator(provider=provider, default_model="gpt-4o-mini")

    from types import MappingProxyType

    request = LLMRequest(
        messages=({"role": "user", "content": "hi"},),
        metadata=MappingProxyType({"generator": "episode"}),
    )
    response = await orch.execute(request)
    assert response.usage.prompt_tokens == 50
    assert response.usage.completion_tokens == 30
    assert response.usage.total == 80

    stats = orch.stats
    assert stats.total_tokens == 80
    assert stats.total_prompt_tokens == 50
    assert stats.total_completion_tokens == 30
    assert stats.requests_by_phase == {"episode": 1}
    assert stats.tokens_by_phase == {"episode": 80}


@pytest.mark.asyncio
async def test_per_phase_tracking():
    """Test that per-phase stats correctly distinguish different generator types."""
    provider = AsyncMock()
    provider.supports_usage_tracking = True
    provider.complete_with_usage = AsyncMock(
        return_value=("response", {"prompt_tokens": 100, "completion_tokens": 50})
    )
    orch = LLMOrchestrator(provider=provider, default_model="gpt-4o-mini")

    from types import MappingProxyType

    # Simulate episode generation (1 call)
    await orch.execute(
        LLMRequest(
            messages=({"role": "user", "content": "hi"},),
            metadata=MappingProxyType({"generator": "episode"}),
        )
    )
    # Simulate semantic prediction (1 call)
    await orch.execute(
        LLMRequest(
            messages=({"role": "user", "content": "predict"},),
            metadata=MappingProxyType({"generator": "semantic_predict"}),
        )
    )
    # Simulate semantic extraction (1 call)
    await orch.execute(
        LLMRequest(
            messages=({"role": "user", "content": "extract"},),
            metadata=MappingProxyType({"generator": "semantic_extract"}),
        )
    )

    stats = orch.stats
    assert stats.total_requests == 3
    assert stats.total_tokens == 450  # 150 * 3
    assert stats.requests_by_phase["episode"] == 1
    assert stats.requests_by_phase["semantic_predict"] == 1
    assert stats.requests_by_phase["semantic_extract"] == 1


@pytest.mark.asyncio
async def test_execute_emits_phase_span_with_usage():
    provider = AsyncMock()
    provider.supports_usage_tracking = True
    provider.complete_with_usage = AsyncMock(
        return_value=("response", {"prompt_tokens": 12, "completion_tokens": 7})
    )
    tracer = _RecordingTracer()
    orch = LLMOrchestrator(
        provider=provider,
        default_model="gpt-test",
        tracer=tracer,
        trace_attributes={"nemori.agent_id": "agent-a"},
    )

    await orch.execute(
        LLMRequest(
            messages=({"role": "user", "content": "hi"},),
            metadata={"generator": "episode", "user_id": "user-a"},
        )
    )

    assert len(tracer.spans) == 1
    name, attributes, span = tracer.spans[0]
    assert name == "nemori.llm.episode"
    assert attributes["openinference.span.kind"] == "CHAIN"
    assert attributes["nemori.agent_id"] == "agent-a"
    assert attributes["nemori.user_id"] == "user-a"
    assert span.attributes["nemori.llm.prompt_tokens"] == 12
    assert span.attributes["nemori.llm.completion_tokens"] == 7
