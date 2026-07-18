"""Async embedding client implementing EmbeddingProvider."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openai import APIConnectionError, AsyncOpenAI

from nemori.domain.exceptions import EmbeddingError

logger = logging.getLogger("nemori")


class AsyncEmbeddingClient:
    """Async embedding generation via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )
        self._model = model

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    async def probe_dimension(self) -> int:
        """Probe actual embedding dimension by sending a test string."""
        return await self.warmup()

    async def warmup(self, timeout: float = 60.0, retries: int = 2) -> int:
        """Warm the embedding backend and return its native dimension."""
        client = self._client.with_options(max_retries=0)
        attempts = retries + 1

        for attempt in range(attempts):
            try:
                response = await asyncio.wait_for(
                    client.embeddings.create(
                        model=self._model,
                        input="dimension probe",
                    ),
                    timeout=timeout,
                )
                return len(response.data[0].embedding)
            except Exception as e:
                if not self._is_transient(e) or attempt == attempts - 1:
                    raise EmbeddingError(
                        f"Embedding warm-up failed after {attempt + 1} attempts: {e}"
                    ) from e
                await asyncio.sleep(2**attempt)

        raise AssertionError("unreachable")

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        current: BaseException | None = error
        while current is not None:
            if isinstance(current, (TimeoutError, APIConnectionError)):
                return True
            status_code = getattr(current, "status_code", None)
            if isinstance(status_code, int) and status_code >= 500:
                return True
            current = current.__cause__
        return False

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"Batch embedding failed: {e}") from e
