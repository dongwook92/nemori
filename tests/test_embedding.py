"""Tests for AsyncEmbeddingClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from nemori.services.embedding import AsyncEmbeddingClient
from nemori.domain.exceptions import EmbeddingError


def test_client_passes_default_headers():
    headers = {"Modal-Key": "modal-key", "Modal-Secret": "modal-secret"}

    with patch("nemori.services.embedding.AsyncOpenAI") as mock_openai:
        AsyncEmbeddingClient(
            api_key="test",
            model="ixi-embedding-v1",
            base_url="https://example.modal.run/v1",
            default_headers=headers,
        )

    mock_openai.assert_called_once_with(
        api_key="test",
        base_url="https://example.modal.run/v1",
        default_headers=headers,
    )


@pytest.mark.asyncio
async def test_embed_returns_float_list():
    client = AsyncEmbeddingClient(api_key="test", model="text-embedding-3-small")
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3]

    with patch.object(client, "_client") as mock_client:
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        result = await client.embed("hello")
        assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_batch_returns_list_of_lists():
    client = AsyncEmbeddingClient(api_key="test", model="text-embedding-3-small")
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]

    with patch.object(client, "_client") as mock_client:
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        result = await client.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_warmup_retries_twice_then_returns_dimension():
    client = AsyncEmbeddingClient(api_key="test", model="ixi-embedding-v1")
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1024)]
    request = AsyncMock(side_effect=[TimeoutError(), TimeoutError(), mock_response])

    with (
        patch.object(client, "_client") as mock_client,
        patch("nemori.services.embedding.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_client.with_options.return_value.embeddings.create = request
        dimension = await client.warmup(timeout=60.0, retries=2)

    assert dimension == 1024
    assert request.await_count == 3
    mock_client.with_options.assert_called_once_with(max_retries=0)


@pytest.mark.asyncio
async def test_warmup_raises_after_two_retries():
    client = AsyncEmbeddingClient(api_key="test", model="ixi-embedding-v1")
    request = AsyncMock(side_effect=TimeoutError())

    with (
        patch.object(client, "_client") as mock_client,
        patch("nemori.services.embedding.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_client.with_options.return_value.embeddings.create = request
        with pytest.raises(EmbeddingError, match="after 3 attempts"):
            await client.warmup(timeout=60.0, retries=2)

    assert request.await_count == 3


@pytest.mark.asyncio
async def test_no_dimensions_param():
    """Verify dimensions parameter is no longer passed to OpenAI API."""
    client = AsyncEmbeddingClient(api_key="test", model="text-embedding-3-small")
    assert not hasattr(client, "_dimensions")
