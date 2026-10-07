"""Exercise the real LangChain RAG pipeline with the configured Gemini key.

Run from the project root: python scripts/live_pipeline_smoke.py
This makes live Gemini requests and may download the local cross encoder.
"""

from pathlib import Path
import sys

from langchain_chroma import Chroma


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from document_rag.chunking import chunk_documents
from document_rag.config import load_config
from document_rag.embeddings import create_embeddings
from document_rag.generation import generate_answer
from document_rag.models import UploadedDocument
from document_rag.parsing import parse_documents
from document_rag.retrieval import retrieve
from document_rag.vector_store import build_index, is_index_ready, reset_index


def main() -> None:
    config = load_config()
    config.require_google_api_key()
    embeddings = create_embeddings(config)
    vector = embeddings.embed_query("annual leave")
    assert len(vector) == config.embedding_dimensions
    print(f"Configuration and Gemini embeddings: OK ({len(vector)} dimensions)")

    first_index = None
    second_index = None
    try:
        uploads = [
            UploadedDocument(
                filename="policy.md",
                content=(ROOT / "evaluation" / "policy.md").read_bytes(),
            ),
            UploadedDocument(
                filename="benefits.txt",
                content=b"Benefits: Employees receive a wellness budget of 500 units per year.",
            ),
        ]
        parsed = parse_documents(uploads, config=config)
        chunks = chunk_documents(parsed, config=config)
        assert {document.source_name for document in parsed} == {"policy.md", "benefits.txt"}
        assert chunks
        print(f"Unstructured parsing and chunking: OK ({len(parsed)} documents, {len(chunks)} chunks)")

        first_index = build_index(chunks, embeddings=embeddings, config=config)
        assert isinstance(first_index.store, Chroma)
        assert first_index.client.get_settings().is_persistent is False
        assert is_index_ready(first_index)
        collection = first_index.client.get_collection(first_index.collection_name)
        assert collection.count() == len(chunks)
        stored = collection.get(ids=[chunks[0].chunk_id])
        assert stored["metadatas"][0]["source_name"] == chunks[0].source_name
        print("LangChain Chroma in-memory index and source metadata: OK")

        question = "How many days of annual leave do employees receive?"
        hits = retrieve(question, first_index, config=config)
        assert hits and any(hit.source == "policy.md" for hit in hits)
        assert all(hit.retrieval_score is not None for hit in hits)
        if config.rerank_enabled:
            assert all(hit.rerank_score is not None for hit in hits), first_index.rerank_warning
        answer = generate_answer(question, hits, config=config)
        assert not answer.insufficient_evidence
        assert "20" in answer.answer
        assert answer.citations
        assert all(citation.chunk_id in {hit.chunk_id for hit in hits} for citation in answer.citations)
        print(f"Hybrid retrieval, reranking, answer, and citations: OK ({len(hits)} passages)")

        if config.query_decomposition_enabled:
            retrieve("Compare annual leave and remote work policies", first_index, config=config)
            assert len(first_index.last_queries) > 1, first_index.query_warning
            print(f"Gemini query decomposition: OK ({len(first_index.last_queries)} search queries)")

        replacement = [UploadedDocument(
            filename="replacement.txt",
            content=b"The replacement policy grants 12 days of annual leave.",
        )]
        replacement_chunks = chunk_documents(parse_documents(replacement, config=config), config=config)
        second_index = build_index(
            replacement_chunks,
            embeddings=embeddings,
            config=config,
            previous_index=first_index,
        )
        assert not is_index_ready(first_index)
        assert first_index.lexical_index.chunks == ()
        assert is_index_ready(second_index)
        new_hits = retrieve("How many days of annual leave?", second_index, config=config)
        assert new_hits and all(hit.source == "replacement.txt" for hit in new_hits)
        assert all("20 days" not in hit.text for hit in new_hits)
        print("Document-set replacement and old/new isolation: OK")
    finally:
        reset_index(first_index)
        reset_index(second_index)
        assert not is_index_ready(first_index)
        assert not is_index_ready(second_index)
        print("In-memory index reset: OK")


if __name__ == "__main__":
    main()
