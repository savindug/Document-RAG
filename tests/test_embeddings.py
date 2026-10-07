"""Gemini adapter configuration without network access."""

import pytest

from document_rag.config import AppConfig
from document_rag.embeddings import create_embeddings


def test_gemini_embedding_factory_uses_configured_model_and_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    import document_rag.embeddings as embedding_module

    seen: dict[str, object] = {}

    def fake_constructor(**kwargs: object) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(embedding_module, "GoogleGenerativeAIEmbeddings", fake_constructor)
    create_embeddings(AppConfig(google_api_key="test-key"))

    assert seen["model"] == "gemini-embedding-2"
    assert seen["output_dimensionality"] == 768
    assert seen["vertexai"] is False
    assert seen["api_key"].get_secret_value() == "test-key"


def test_gemini_embedding_factory_requires_key() -> None:
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        create_embeddings(AppConfig())
