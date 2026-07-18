"""Tests for tenant-scoped Qdrant operations."""

from unittest.mock import MagicMock, patch

from qdrant_client.models import FieldCondition, HasIdCondition

from nemori.db.qdrant_store import QdrantVectorStore


def _store_with_mock_client() -> tuple[QdrantVectorStore, MagicMock]:
    with patch("nemori.db.qdrant_store.QdrantClient") as client_cls:
        store = QdrantVectorStore()
    return store, client_cls.return_value


def _assert_tenant_delete(selector, record_id: str) -> None:
    conditions = selector.must
    assert conditions is not None
    assert HasIdCondition(has_id=[record_id]) in conditions
    assert FieldCondition(key="user_id", match={"value": "user-a"}) in conditions
    assert FieldCondition(key="agent_id", match={"value": "agent-a"}) in conditions


def test_delete_episode_filters_by_id_and_both_tenant_keys():
    store, client = _store_with_mock_client()

    store.delete_episode("episode-a", "user-a", "agent-a")

    kwargs = client.delete.call_args.kwargs
    assert kwargs["collection_name"] == "nemori_episodes"
    _assert_tenant_delete(kwargs["points_selector"], "episode-a")


def test_delete_semantic_filters_by_id_and_both_tenant_keys():
    store, client = _store_with_mock_client()

    store.delete_semantic("memory-a", "user-a", "agent-a")

    kwargs = client.delete.call_args.kwargs
    assert kwargs["collection_name"] == "nemori_semantic"
    _assert_tenant_delete(kwargs["points_selector"], "memory-a")
