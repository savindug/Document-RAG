"""Ephemeral Chroma collection lifecycle and dataset-scoped vector search."""

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from document_rag.config import AppConfig
from document_rag.embeddings import create_embeddings
from document_rag.lexical import LexicalIndex, build_lexical_index
from document_rag.models import DocumentChunk, RetrievedChunk


class IndexNotReadyError(ValueError):
    """No queryable collection is available for this document set."""


@dataclass
class VectorIndex:
    """A single active document set and its in-memory Chroma collection."""

    client: ClientAPI
    store: Chroma
    collection_name: str
    chunk_count: int
    retrieval_top_k: int
    lexical_index: LexicalIndex
    rerank_warning: str | None = None
    query_warning: str | None = None
    last_queries: tuple[str, ...] = ()
    ready: bool = True


def _metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "source_name": chunk.source_name,
        "text": chunk.text,
    }
    if chunk.page_number is not None:
        metadata["page_number"] = chunk.page_number
    if chunk.heading is not None:
        metadata["heading"] = chunk.heading
    if chunk.element_type is not None:
        metadata["element_type"] = chunk.element_type
    return metadata


def _delete_collection(client: ClientAPI, name: str) -> None:
    try:
        client.delete_collection(name)
    except NotFoundError:
        pass


def build_index(
    chunks: Iterable[DocumentChunk],
    *,
    embeddings: Embeddings | None = None,
    config: AppConfig | None = None,
    previous_index: VectorIndex | None = None,
    client: ClientAPI | None = None,
) -> VectorIndex:
    """Build a fresh collection, then retire the previous one after success."""
    settings = config or AppConfig()
    items = list(chunks)
    if not items:
        raise ValueError("Cannot build an index without chunks.")
    if len(items) > settings.max_chunks:
        raise ValueError(f"Document set exceeds the {settings.max_chunks} chunk limit.")
    if len({chunk.chunk_id for chunk in items}) != len(items):
        raise ValueError("Chunk IDs must be unique within an index.")
    if any(not chunk.embedding_text.strip() for chunk in items):
        raise ValueError("Cannot index a blank chunk.")

    lexical_index = build_lexical_index(items)
    embedding_model = embeddings or create_embeddings(config)
    active_client = client or chromadb.EphemeralClient()
    name = f"rag_{uuid4().hex}"
    try:
        store = Chroma(
            client=active_client,
            collection_name=name,
            embedding_function=embedding_model,
            collection_configuration={"hnsw": {"space": "cosine"}},
        )
        documents = [Document(page_content=chunk.embedding_text, metadata=_metadata(chunk)) for chunk in items]
        store.add_documents(documents, ids=[chunk.chunk_id for chunk in items])
        count = active_client.get_collection(name).count()
        if count != len(items):
            raise RuntimeError(f"Index contains {count} chunks; expected {len(items)}.")
    except Exception:
        _delete_collection(active_client, name)
        raise

    index = VectorIndex(
        client=active_client,
        store=store,
        collection_name=name,
        chunk_count=len(items),
        retrieval_top_k=settings.retrieval_top_k,
        lexical_index=lexical_index,
    )
    if previous_index is not None:
        try:
            reset_index(previous_index)
        except Exception:
            _delete_collection(active_client, name)
            index.ready = False
            raise
    return index


def is_index_ready(index: VectorIndex | None) -> bool:
    """Check both lifecycle state and the actual Chroma collection count."""
    if index is None or not index.ready:
        return False
    try:
        return index.client.get_collection(index.collection_name).count() == index.chunk_count
    except NotFoundError:
        return False


def search_index(
    index: VectorIndex | None, query: str, *, top_k: int | None = None
) -> list[RetrievedChunk]:
    """Search only the collection owned by this index and return source-rich hits."""
    if not is_index_ready(index):
        raise IndexNotReadyError("Build an index before searching.")
    if not query.strip():
        raise ValueError("Question must not be blank.")
    assert index is not None
    limit = top_k if top_k is not None else index.retrieval_top_k
    if limit < 1:
        raise ValueError("top_k must be at least 1.")

    results = index.store.similarity_search_with_score(query, k=min(limit, index.chunk_count))
    hits: list[RetrievedChunk] = []
    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        chunk = DocumentChunk(
            chunk_id=str(metadata["chunk_id"]),
            source_id=str(metadata["source_id"]),
            source_name=str(metadata["source_name"]),
            text=str(metadata["text"]),
            embedding_text=document.page_content,
            page_number=metadata.get("page_number"),
            heading=metadata.get("heading"),
            element_type=metadata.get("element_type"),
        )
        hits.append(RetrievedChunk(chunk=chunk, rank=rank, distance=max(0.0, float(distance))))
    return hits


def reset_index(index: VectorIndex | None) -> None:
    """Delete this index's collection and mark it unavailable; safe to repeat."""
    if index is None:
        return
    _delete_collection(index.client, index.collection_name)
    index.ready = False
    index.lexical_index = build_lexical_index([])
    index.rerank_warning = None
    index.query_warning = None
    index.last_queries = ()
