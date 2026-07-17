"""Async OpenAI-compatible LLM client implementing LLMProvider."""
from __future__ import annotations

import logging
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from nemori.domain.exceptions import LLMError, LLMAuthError, LLMRateLimitError

logger = logging.getLogger("nemori")


class AsyncLLMClient:
    """Async LLM client wrapping the OpenAI API."""

    supports_usage_tracking: bool = True

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, messages: list[dict], **kwargs: Any) -> str:
        content, _ = await self.complete_with_usage(messages, **kwargs)
        return content

    async def complete_with_usage(
        self, messages: list[dict], **kwargs: Any
    ) -> tuple[str, dict[str, int]]:
        """Return (content, {"prompt_tokens": ..., "completion_tokens": ...})."""
        model = kwargs.pop("model", "qwen3.6-27b-fp8")
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", 2000)
        typed_messages = cast(list[ChatCompletionMessageParam], messages)
        if model == "qwen3.6-27b-fp8":
            kwargs.setdefault("top_p", 0.8)
            kwargs.setdefault("presence_penalty", 1.5)
            kwargs.setdefault(
                "extra_body",
                {
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        token_limit = (
            {"max_completion_tokens": max_tokens}
            if model.startswith("gpt-5")
            else {"max_tokens": max_tokens}
        )

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=typed_messages,
                temperature=temperature,
                **token_limit,
                **kwargs,
            )
            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            }
            return content, usage
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "403" in error_str:
                raise LLMAuthError(f"Authentication failed: {e}") from e
            if "429" in error_str:
                raise LLMRateLimitError(f"Rate limited: {e}") from e
            raise LLMError(f"LLM call failed: {e}") from e
