"""Configuration contract checks that never contact Gemini."""

import pytest
from pathlib import Path

import document_rag.config as config_module
from document_rag.config import AppConfig, load_config


def test_default_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", Path(__file__).parent / "missing_env")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("UNSTRUCTURED_API_KEY", raising=False)
    monkeypatch.delenv("UNSTRUCTURED_API_URL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
    config = load_config()
    assert config.gemini_model == "gemini-3.5-flash-lite"
    assert config.gemini_embedding_model == "gemini-embedding-2"
    assert config.embedding_dimensions == 768
    assert config.unstructured_api_key is None
    assert config.unstructured_api_url is None


def test_missing_key_has_setup_hint() -> None:
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        AppConfig().require_google_api_key()
    assert AppConfig(google_api_key="  ").google_api_key is None


def test_top_k_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "7")
    assert load_config().retrieval_top_k == 7


def test_unstructured_key_is_loaded_as_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", Path(__file__).parent / "missing_env")
    monkeypatch.delenv("UNSTRUCTURED_API_URL", raising=False)
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "test-partition-key")
    settings = load_config()
    assert settings.unstructured_api_key.get_secret_value() == "test-partition-key"
    assert "test-partition-key" not in repr(settings)


def test_transform_url_is_not_a_partition_endpoint() -> None:
    with pytest.raises(ValueError, match="Transform URLs are not compatible"):
        AppConfig(unstructured_api_url="https://transform.unstructured.io/api/v2")


def test_invalid_environment_url_falls_back_to_local_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", Path(__file__).parent / "missing_env")
    monkeypatch.setenv("UNSTRUCTURED_API_URL", "https://transform.unstructured.io/api/v2")
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "test-key")
    settings = load_config()
    assert settings.unstructured_api_url is None
    assert settings.unstructured_api_key is not None
    assert config_module.unstructured_api_url_warning(
        "https://transform.unstructured.io/api/v2"
    ) is not None
