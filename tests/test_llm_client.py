"""Tests for AsyncLLMClient."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nemori.llm.client import AsyncLLMClient


@pytest.mark.asyncio
async def test_complete_returns_string():
    client = AsyncLLMClient(api_key="test-key", base_url=None)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello!"

    with patch.object(client, "_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await client.complete([{"role": "user", "content": "hi"}])
        assert result == "Hello!"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "qwen3.6-27b-fp8"
        assert call_kwargs["max_tokens"] == 2000
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["presence_penalty"] == 1.5
        assert call_kwargs["extra_body"] == {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        assert "max_completion_tokens" not in call_kwargs


@pytest.mark.asyncio
async def test_complete_passes_params():
    client = AsyncLLMClient(api_key="test-key", base_url=None)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"

    with patch.object(client, "_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        await client.complete(
            [{"role": "user", "content": "hi"}],
            model="gpt-4o", temperature=0.5, max_tokens=100,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_complete_uses_max_completion_tokens_for_gpt5():
    client = AsyncLLMClient(api_key="test-key", base_url=None)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"

    with patch.object(client, "_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        await client.complete(
            [{"role": "user", "content": "hi"}],
            model="gpt-5.4-nano",
            max_tokens=100,
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 100
        assert "max_tokens" not in call_kwargs
