"""Credential-free tests for the question-to-evidence boundary."""

import pytest
from langchain_core.embeddings import Embeddings

from document_rag.config import AppConfig
from document_rag.models import DocumentChunk
from document_rag.retrieval import retrieve
from document_rag.vector_store import IndexNotReadyError, build_index, reset_index, search_index


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lower = text.lower()
        return [1.0 if "alpha" in lower else 0.01, 1.0 if "beta" in lower else 0.01]


@pytest.fixture(autouse=True)
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "document_rag.retrieval.rerank_scores",
        lambda model, question, passages: [0.0 for _ in passages],
    )


def chunk(number: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{number}",
        source_id="source-1",
        source_name="report.pdf",
        text=text,
        embedding_text=text,
        page_number=3,
        heading="Key findings",
        element_type="NarrativeText",
    )


def test_retrieve_returns_ranked_source_attributed_results() -> None:
    index = build_index(
        [chunk(1, "Alpha findings"), chunk(2, "Beta findings")],
        embeddings=KeywordEmbeddings(),
    )
    try:
        results = retrieve("  alpha  ", index, top_k=1)

        assert len(results) == 1
        assert results[0].model_dump() == {
            "rank": 1,
            "chunk_id": "chunk-1",
            "text": "Alpha findings",
            "source": "report.pdf",
            "page": 3,
            "section": "Key findings",
            "element_type": "NarrativeText",
            "distance": results[0].distance,
            "lexical_score": results[0].lexical_score,
            "retrieval_score": results[0].retrieval_score,
            "rerank_score": 0.0,
        }
        assert results[0].distance is not None
        assert results[0].distance >= 0
    finally:
        reset_index(index)


def test_lexical_candidate_can_rescue_a_vector_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    index = build_index(
        [chunk(1, "Alpha general overview"), chunk(2, "Zebra-specific leave rules")],
        embeddings=KeywordEmbeddings(),
    )
    try:
        vector_hit = search_index(index, "alpha", top_k=1)[0]
        monkeypatch.setattr("document_rag.retrieval.search_index", lambda *args, **kwargs: [vector_hit])
        results = retrieve(
            "zebra",
            index,
            top_k=2,
            config=AppConfig(retrieval_candidate_pool=1, rerank_enabled=False),
        )
        assert {result.chunk_id for result in results} == {"chunk-1", "chunk-2"}
        assert next(result for result in results if result.chunk_id == "chunk-2").lexical_score > 0
    finally:
        reset_index(index)


def test_cross_encoder_reranks_fused_candidates() -> None:
    index = build_index(
        [chunk(1, "Alpha general overview"), chunk(2, "Alpha leave rule")],
        embeddings=KeywordEmbeddings(),
    )
    seen = []

    def rank(model: str, question: str, passages: list[str]) -> list[float]:
        seen.extend(passages)
        return [1.0 if "leave rule" in passage else 0.0 for passage in passages]

    try:
        results = retrieve("alpha", index, top_k=1, reranker=rank)
        assert len(seen) == 2
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-2"
        assert results[0].rerank_score == 1.0
    finally:
        reset_index(index)


def test_reranker_failure_keeps_hybrid_results() -> None:
    index = build_index([chunk(1, "Alpha fact")], embeddings=KeywordEmbeddings())

    def unavailable(*args):
        raise OSError("model unavailable")

    try:
        results = retrieve("alpha", index, reranker=unavailable)
        assert results[0].chunk_id == "chunk-1"
        assert results[0].rerank_score is None
        assert "unavailable" in index.rerank_warning
    finally:
        reset_index(index)


def test_multipart_question_searches_subqueries_and_keeps_top_ranked_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = build_index(
        [chunk(1, "Alpha annual leave"), chunk(2, "Beta sick leave")],
        embeddings=KeywordEmbeddings(),
    )
    try:
        alpha_hit = search_index(index, "alpha", top_k=1)[0]
        beta_hit = search_index(index, "beta", top_k=1)[0]
        searched = []

        def search(active_index, query, *, top_k):
            searched.append(query)
            return [beta_hit if query == "beta sick leave" else alpha_hit]

        monkeypatch.setattr("document_rag.retrieval.search_index", search)
        question = "Compare alpha annual leave and beta sick leave"
        results = retrieve(
            question,
            index,
            top_k=1,
            config=AppConfig(retrieval_candidate_pool=1),
            query_planner=lambda q, settings, maximum: [q, "alpha annual leave", "beta sick leave"],
            reranker=lambda model, q, passages: [1.0 if "sick leave" in passage else 0.0 for passage in passages],
        )
        assert searched == [question, "alpha annual leave", "beta sick leave"]
        assert index.last_queries == tuple(searched)
        assert results[0].chunk_id == "chunk-2"
        assert results[0].rerank_score == 1.0
    finally:
        reset_index(index)


def test_query_planning_failure_searches_original_question() -> None:
    index = build_index([chunk(1, "Alpha and beta")], embeddings=KeywordEmbeddings())

    def fail(*args):
        raise RuntimeError("planner unavailable")

    try:
        results = retrieve("Alpha and beta?", index, query_planner=fail)
        assert results[0].chunk_id == "chunk-1"
        assert index.last_queries == ("Alpha and beta?",)
        assert "unavailable" in index.query_warning
    finally:
        reset_index(index)


def test_top_k_defaults_to_five_and_can_be_overridden() -> None:
    index = build_index([chunk(i, f"Alpha fact {i}") for i in range(7)], embeddings=KeywordEmbeddings())
    try:
        assert len(retrieve("alpha", index, config=AppConfig())) == 5
        assert len(retrieve("alpha", index, top_k=2, config=AppConfig())) == 2
        assert len(retrieve("alpha", index, config=AppConfig(retrieval_top_k=3))) == 3
    finally:
        reset_index(index)


def test_top_k_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "1")
    index = build_index([chunk(1, "Alpha"), chunk(2, "Beta")], embeddings=KeywordEmbeddings())
    try:
        assert len(retrieve("alpha", index)) == 1
    finally:
        reset_index(index)


def test_rejects_blank_question_and_missing_or_reset_index() -> None:
    with pytest.raises(ValueError, match="Question must not be blank"):
        retrieve("  \n  ", None)
    with pytest.raises(IndexNotReadyError, match="No active index"):
        retrieve("Where is the evidence?", None)

    index = build_index([chunk(1, "Alpha")], embeddings=KeywordEmbeddings())
    reset_index(index)
    with pytest.raises(IndexNotReadyError, match="No active index"):
        retrieve("Where is the evidence?", index)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_rejects_invalid_top_k(top_k: object) -> None:
    index = build_index([chunk(1, "Alpha")], embeddings=KeywordEmbeddings())
    try:
        with pytest.raises(ValueError, match="TOP_K"):
            retrieve("alpha", index, top_k=top_k)
    finally:
        reset_index(index)
