"""Real ephemeral Chroma tests with deterministic, credential-free embeddings."""

import pytest
from langchain_core.embeddings import Embeddings

from document_rag.config import AppConfig
from document_rag.lexical import search_lexical
from document_rag.models import DocumentChunk
from document_rag.vector_store import (
    IndexNotReadyError,
    build_index,
    is_index_ready,
    reset_index,
    search_index,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lower = text.lower()
        return [1.0 if "alpha" in lower else 0.01, 1.0 if "beta" in lower else 0.01]


class FailingEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("fake embedding failure")


def chunk(chunk_id: str, source: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source,
        source_name=f"{source}.md",
        text=text,
        embedding_text=text,
        page_number=2,
        heading="Overview",
        element_type="NarrativeText",
    )


def test_build_search_metadata_and_reset() -> None:
    index = build_index(
        [chunk("alpha:0", "alpha", "Alpha project notes"), chunk("beta:0", "beta", "Beta project notes")],
        embeddings=FakeEmbeddings(),
    )
    try:
        assert is_index_ready(index)
        assert index.client.get_settings().is_persistent is False
        stored = index.client.get_collection(index.collection_name).get(ids=["alpha:0"])
        metadata = stored["metadatas"][0]
        assert metadata["chunk_id"] == "alpha:0"
        assert metadata["source_name"] == "alpha.md"
        assert metadata["page_number"] == 2
        assert metadata["heading"] == "Overview"
        assert metadata["element_type"] == "NarrativeText"

        hits = search_index(index, "alpha", top_k=1)
        assert len(hits) == 1
        assert hits[0].chunk.chunk_id == "alpha:0"
        assert hits[0].chunk.source_name == "alpha.md"
        assert hits[0].chunk.page_number == 2
        assert hits[0].rank == 1
    finally:
        reset_index(index)

    assert not is_index_ready(index)
    assert index.lexical_index.chunks == ()
    reset_index(index)
    with pytest.raises(IndexNotReadyError):
        search_index(index, "alpha")


def test_new_document_set_is_isolated_and_retires_previous_index() -> None:
    first = build_index([chunk("old:0", "old", "Alpha only")], embeddings=FakeEmbeddings())
    second = None
    try:
        second = build_index(
            [chunk("new:0", "new", "Beta only")],
            embeddings=FakeEmbeddings(),
            previous_index=first,
        )
        assert second.collection_name != first.collection_name
        assert not is_index_ready(first)
        assert is_index_ready(second)
        assert search_lexical(first.lexical_index, "alpha", 5) == []
        assert search_lexical(second.lexical_index, "alpha", 5) == []
        assert search_lexical(second.lexical_index, "beta", 5)[0][0].chunk_id == "new:0"
        assert [hit.chunk.chunk_id for hit in search_index(second, "alpha", top_k=5)] == ["new:0"]
        with pytest.raises(IndexNotReadyError):
            search_index(first, "alpha")
    finally:
        reset_index(first)
        reset_index(second)


def test_two_active_indexes_never_share_documents() -> None:
    first = build_index([chunk("set-a:0", "set-a", "Alpha only")], embeddings=FakeEmbeddings())
    second = build_index([chunk("set-b:0", "set-b", "Beta only")], embeddings=FakeEmbeddings())
    try:
        assert first.collection_name != second.collection_name
        assert search_index(first, "beta", top_k=5)[0].chunk.chunk_id == "set-a:0"
        assert search_index(second, "alpha", top_k=5)[0].chunk.chunk_id == "set-b:0"
    finally:
        reset_index(first)
        reset_index(second)


def test_failed_build_preserves_previous_index_and_cleans_candidate() -> None:
    first = build_index([chunk("old:1", "old", "Alpha remains")], embeddings=FakeEmbeddings())
    names_before = {collection.name for collection in first.client.list_collections()}
    try:
        with pytest.raises(RuntimeError, match="fake embedding failure"):
            build_index(
                [chunk("new:1", "new", "Beta candidate")],
                embeddings=FailingEmbeddings(),
                previous_index=first,
                client=first.client,
            )
        assert is_index_ready(first)
        assert {collection.name for collection in first.client.list_collections()} == names_before
        assert search_index(first, "alpha", top_k=1)[0].chunk.chunk_id == "old:1"
    finally:
        reset_index(first)


def test_validation_happens_before_collection_creation() -> None:
    with pytest.raises(ValueError, match="without chunks"):
        build_index([], embeddings=FakeEmbeddings())
    with pytest.raises(ValueError, match="unique"):
        build_index([chunk("same", "a", "Alpha"), chunk("same", "b", "Beta")], embeddings=FakeEmbeddings())
    with pytest.raises(ValueError, match="blank"):
        build_index([chunk("x", "a", "Alpha").model_copy(update={"embedding_text": " "})], embeddings=FakeEmbeddings())
    with pytest.raises(ValueError, match="Question must not be blank"):
        index = build_index([chunk("x", "a", "Alpha")], embeddings=FakeEmbeddings())
        try:
            search_index(index, " ")
        finally:
            reset_index(index)


def test_fake_embeddings_skip_gemini_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    import document_rag.vector_store as vector_store

    monkeypatch.setattr(vector_store, "create_embeddings", lambda config: (_ for _ in ()).throw(AssertionError("Gemini called")))
    index = build_index([chunk("x", "a", "Alpha")], embeddings=FakeEmbeddings(), config=AppConfig())
    reset_index(index)
