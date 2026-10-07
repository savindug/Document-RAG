"""Hybrid BM25 and vector retrieval over the active document set."""

from collections.abc import Callable
from math import isfinite

from document_rag.config import AppConfig, load_config
from document_rag.lexical import search_lexical
from document_rag.models import DocumentChunk, RetrievalResult
from document_rag.query_planning import looks_composite, plan_queries
from document_rag.reranking import rerank_scores
from document_rag.vector_store import IndexNotReadyError, VectorIndex, is_index_ready, search_index


_RRF_OFFSET = 60


def retrieve(
    question: str,
    index: VectorIndex | None,
    *,
    top_k: int | None = None,
    config: AppConfig | None = None,
    reranker: Callable[[str, str, list[str]], list[float]] | None = None,
    query_planner: Callable[[str, AppConfig, int], list[str]] | None = None,
) -> list[RetrievalResult]:
    """Fuse vector and lexical candidates, then rerank before generating an answer."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be blank.")
    if not is_index_ready(index):
        raise IndexNotReadyError("No active index. Upload documents and build an index first.")

    settings = config or load_config()
    limit = top_k if top_k is not None else settings.retrieval_top_k
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("TOP_K must be a positive integer.")

    assert index is not None
    query = question.strip()
    index.query_warning = None
    queries = [query]
    if settings.query_decomposition_enabled and looks_composite(query):
        try:
            planned = (query_planner or plan_queries)(query, settings, settings.max_subqueries)
            queries.extend(planned[1:settings.max_subqueries + 1])
        except Exception:
            index.query_warning = "Query decomposition was unavailable; searching the original question."
    queries = list(dict.fromkeys(item.strip() for item in queries if isinstance(item, str) and item.strip()))
    index.last_queries = tuple(queries)
    candidate_limit = min(
        index.chunk_count,
        max(limit, settings.retrieval_candidate_pool // len(queries)),
    )

    candidates: dict[str, dict[str, object]] = {}
    for query_number, search_query in enumerate(queries):
        weight = 1.0 if query_number == 0 else 0.8
        vector_hits = search_index(index, search_query, top_k=candidate_limit)
        lexical_hits = search_lexical(index.lexical_index, search_query, candidate_limit)
        for rank, hit in enumerate(vector_hits, start=1):
            candidate = candidates.setdefault(
                hit.chunk.chunk_id,
                {"chunk": hit.chunk, "distance": None, "lexical_score": None, "retrieval_score": 0.0},
            )
            old_distance = candidate["distance"]
            candidate["distance"] = (
                min(old_distance, hit.distance) if old_distance is not None else hit.distance
            )
            candidate["retrieval_score"] = float(candidate["retrieval_score"]) + weight / (_RRF_OFFSET + rank)
        for rank, (chunk, lexical_score) in enumerate(lexical_hits, start=1):
            candidate = candidates.setdefault(
                chunk.chunk_id,
                {"chunk": chunk, "distance": None, "lexical_score": None, "retrieval_score": 0.0},
            )
            old_lexical = candidate["lexical_score"]
            candidate["lexical_score"] = (
                max(old_lexical, lexical_score) if old_lexical is not None else lexical_score
            )
            candidate["retrieval_score"] = float(candidate["retrieval_score"]) + weight / (_RRF_OFFSET + rank)

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (-float(candidate["retrieval_score"]), candidate["chunk"].chunk_id),
    )
    index.rerank_warning = None
    if settings.rerank_enabled and ordered:
        score_fn = reranker or rerank_scores
        try:
            scores = score_fn(
                settings.rerank_model,
                query,
                [candidate["chunk"].embedding_text for candidate in ordered],
            )
            if len(scores) != len(ordered) or any(not isfinite(float(score)) for score in scores):
                raise ValueError("Cross-encoder returned invalid scores.")
            for candidate, score in zip(ordered, scores):
                candidate["rerank_score"] = float(score)
            ordered.sort(
                key=lambda candidate: (
                    -float(candidate["rerank_score"]),
                    -float(candidate["retrieval_score"]),
                    candidate["chunk"].chunk_id,
                )
            )
        except Exception:
            index.rerank_warning = (
                "Cross-encoder reranking is unavailable; results use hybrid ranking. "
                "Check the local reranker model download and installation."
            )

    results: list[RetrievalResult] = []
    for rank, candidate in enumerate(ordered[:limit], start=1):
        chunk: DocumentChunk = candidate["chunk"]
        results.append(
            RetrievalResult(
                rank=rank,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source_name,
                page=chunk.page_number,
                section=chunk.heading,
                element_type=chunk.element_type,
                distance=candidate["distance"],
                lexical_score=candidate["lexical_score"],
                retrieval_score=candidate["retrieval_score"],
                rerank_score=candidate.get("rerank_score"),
            )
        )
    return results
