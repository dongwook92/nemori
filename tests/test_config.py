"""Tests for simplified MemoryConfig."""

import pytest
from nemori.config import MemoryConfig
from nemori.domain.exceptions import ConfigError


def test_default_config():
    cfg = MemoryConfig()
    assert "nemori" in cfg.dsn  # DSN resolved from env or default
    assert cfg.db_pool_min == 5
    assert cfg.db_pool_max == 20
    assert cfg.agent_id == "default"
    assert cfg.llm_model == "qwen3.6-27b-fp8"
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.embedding_dimension == 1536
    assert cfg.embedding_warmup_timeout == 60.0
    assert cfg.embedding_warmup_retries == 2
    assert cfg.enable_llm_tracing is False
    assert cfg.phoenix_collector_endpoint == "http://localhost:6006/v1/traces"
    assert cfg.phoenix_project_name == "nemori"
    assert cfg.trace_hide_inputs is True
    assert cfg.trace_hide_outputs is True
    assert cfg.trace_hide_embeddings_vectors is True
    assert cfg.trace_hide_embeddings_text is True
    assert cfg.buffer_size_min == 2
    assert cfg.search_top_k_episodes == 10


def test_config_reads_env_for_llm_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key-123")
    cfg = MemoryConfig()
    assert cfg.llm_api_key == "test-key-123"


def test_config_repr_hides_credentials():
    cfg = MemoryConfig(
        dsn="postgresql://user:db-secret@localhost/nemori",
        llm_api_key="llm-secret",
        embedding_api_key="embedding-secret",
        qdrant_url="https://user:qdrant-url-secret@qdrant.example.com",
        qdrant_api_key="qdrant-secret",
        phoenix_api_key="phoenix-secret",
        embedding_base_url="https://example.modal.run/v1",
        embedding_headers={"Modal-Secret": "modal-secret"},
    )

    rendered = repr(cfg)
    for secret in (
        "db-secret",
        "llm-secret",
        "embedding-secret",
        "qdrant-url-secret",
        "qdrant-secret",
        "phoenix-secret",
        "modal-secret",
    ):
        assert secret not in rendered


def test_config_reads_phoenix_tracing_env(monkeypatch):
    monkeypatch.setenv("NEMORI_ENABLE_LLM_TRACING", "true")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "nemori-e2e")

    cfg = MemoryConfig()

    assert cfg.enable_llm_tracing is True
    assert cfg.phoenix_collector_endpoint == "http://phoenix:6006/v1/traces"
    assert cfg.phoenix_project_name == "nemori-e2e"


def test_config_rejects_invalid_tracing_boolean(monkeypatch):
    monkeypatch.setenv("NEMORI_ENABLE_LLM_TRACING", "sometimes")
    with pytest.raises(ConfigError, match="NEMORI_ENABLE_LLM_TRACING"):
        MemoryConfig()


def test_config_reads_embedding_model_and_modal_headers(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "ixi-embedding-v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.modal.run/v1")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_ID", "modal-key")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_SECRET", "modal-secret")

    cfg = MemoryConfig()

    assert cfg.embedding_model == "ixi-embedding-v1"
    assert cfg.embedding_headers == {
        "Modal-Key": "modal-key",
        "Modal-Secret": "modal-secret",
    }


def test_config_does_not_send_modal_headers_to_other_hosts(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_ID", "modal-key")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_SECRET", "modal-secret")

    assert MemoryConfig().embedding_headers is None


def test_config_does_not_send_modal_headers_over_http(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://example.modal.run/v1")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_ID", "modal-key")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_SECRET", "modal-secret")

    assert MemoryConfig().embedding_headers is None


def test_config_rejects_modal_headers_when_constructor_overrides_host(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.modal.run/v1")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_ID", "modal-key")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_SECRET", "modal-secret")

    with pytest.raises(ConfigError, match="Modal auth headers"):
        MemoryConfig(embedding_base_url="https://attacker.example/v1")


def test_config_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fallback")
    cfg = MemoryConfig()
    assert cfg.llm_api_key == "openai-fallback"


def test_config_custom_dsn():
    cfg = MemoryConfig(dsn="postgresql://user:pass@db:5432/mydb")
    assert cfg.dsn == "postgresql://user:pass@db:5432/mydb"


def test_config_custom_agent_id():
    cfg = MemoryConfig(agent_id="my-agent")
    assert cfg.agent_id == "my-agent"


def test_config_no_removed_fields():
    cfg = MemoryConfig()
    assert not hasattr(cfg, "storage_backend")
    assert not hasattr(cfg, "vector_index_backend")
    assert not hasattr(cfg, "lexical_index_backend")
    assert not hasattr(cfg, "chroma_persist_directory")
    assert not hasattr(cfg, "storage_path")
    assert hasattr(cfg, "enable_episode_merging")
    assert not hasattr(cfg, "max_workers")


def test_config_invalid_db_pool_min():
    with pytest.raises(ConfigError, match="db_pool_min"):
        MemoryConfig(db_pool_min=0)


def test_config_invalid_db_pool_max():
    with pytest.raises(ConfigError, match="db_pool_max"):
        MemoryConfig(db_pool_min=10, db_pool_max=5)


def test_config_invalid_buffer_size_min():
    with pytest.raises(ConfigError, match="buffer_size_min"):
        MemoryConfig(buffer_size_min=0)


def test_config_invalid_buffer_size_max():
    with pytest.raises(ConfigError, match="buffer_size_max"):
        MemoryConfig(buffer_size_min=10, buffer_size_max=5)


def test_config_has_qdrant_defaults():
    cfg = MemoryConfig()
    assert cfg.qdrant_url == "localhost"
    assert cfg.qdrant_port == 6333
    assert cfg.qdrant_api_key is None
    assert cfg.qdrant_collection_prefix == "nemori"


def test_config_invalid_search_top_k_episodes():
    with pytest.raises(ConfigError, match="search_top_k_episodes"):
        MemoryConfig(search_top_k_episodes=0)


def test_config_invalid_search_top_k_semantic():
    with pytest.raises(ConfigError, match="search_top_k_semantic"):
        MemoryConfig(search_top_k_semantic=0)


def test_config_invalid_embedding_warmup_timeout():
    with pytest.raises(ConfigError, match="embedding_warmup_timeout"):
        MemoryConfig(embedding_warmup_timeout=0)


def test_config_invalid_embedding_warmup_retries():
    with pytest.raises(ConfigError, match="embedding_warmup_retries"):
        MemoryConfig(embedding_warmup_retries=-1)


def test_config_rejects_plaintext_public_trace_endpoint():
    with pytest.raises(ConfigError, match="phoenix_collector_endpoint.*HTTPS"):
        MemoryConfig(
            enable_llm_tracing=True,
            phoenix_collector_endpoint="http://collector.example.com/v1/traces",
        )


def test_config_allows_explicit_http_sidecar_endpoints():
    cfg = MemoryConfig(
        llm_base_url="http://api.example.com/v1",
        embedding_base_url="http://embeddings.example.com/v1",
    )

    assert cfg.llm_base_url == "http://api.example.com/v1"


def test_config_rejects_credentials_embedded_in_endpoint():
    with pytest.raises(ConfigError, match="llm_base_url.*credentials"):
        MemoryConfig(llm_base_url="https://user:secret@api.example.com/v1")


def test_config_allows_plaintext_local_service_endpoints():
    cfg = MemoryConfig(
        llm_base_url="http://llm-sidecar:8000/v1",
        embedding_base_url="http://127.0.0.1:8001/v1",
        enable_llm_tracing=True,
        phoenix_collector_endpoint="http://phoenix:6006/v1/traces",
    )

    assert cfg.llm_base_url == "http://llm-sidecar:8000/v1"
