"""Streamlit interface for a session-local document RAG index."""

from hashlib import sha256
import os
from typing import Any

import streamlit as st
from pydantic import ValidationError

from document_rag.chunking import chunk_documents
from document_rag.config import load_config, unstructured_api_url_warning
from document_rag.generation import generate_answer
from document_rag.models import UploadedDocument
from document_rag.parsing import parse_documents
from document_rag.retrieval import retrieve
from document_rag.vector_store import build_index, is_index_ready, reset_index


UPLOAD_TYPES = ["pdf", "txt", "md", "markdown", "doc", "docx", "csv"]


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "active_index": None,
        "active_signature": None,
        "active_filenames": [],
        "parsed_documents": [],
        "chunks": [],
        "last_retrieved": [],
        "last_answer": None,
        "last_question": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _selection_signature(files: list[Any]) -> str | None:
    if not files:
        return None
    digest = sha256()
    for file in files:
        name = file.name.encode("utf-8")
        content = file.getvalue()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _clear_active_state() -> None:
    st.session_state.active_index = None
    st.session_state.active_signature = None
    st.session_state.active_filenames = []
    st.session_state.parsed_documents = []
    st.session_state.chunks = []
    st.session_state.last_retrieved = []
    st.session_state.last_answer = None
    st.session_state.last_question = None


def _build(files: list[Any], signature: str) -> None:
    settings = load_config()
    settings.require_google_api_key()
    uploads = [UploadedDocument(filename=file.name, content=file.getvalue()) for file in files]
    parsed = parse_documents(uploads, config=settings)
    chunks = chunk_documents(parsed, config=settings)
    index = build_index(chunks, config=settings, previous_index=st.session_state.active_index)
    st.session_state.active_index = index
    st.session_state.active_signature = signature
    st.session_state.active_filenames = [file.name for file in files]
    st.session_state.parsed_documents = parsed
    st.session_state.chunks = chunks
    st.session_state.last_retrieved = []
    st.session_state.last_answer = None
    st.session_state.last_question = None


def _ask(question: str) -> None:
    settings = load_config()
    st.session_state.last_retrieved = []
    st.session_state.last_answer = None
    st.session_state.last_question = None
    results = retrieve(question, st.session_state.active_index, config=settings)
    st.session_state.last_retrieved = results
    st.session_state.last_answer = generate_answer(question, results, config=settings)
    st.session_state.last_question = question


def _validation_message(error: ValidationError) -> str:
    """Show validation messages without echoing uploaded bytes from Pydantic input."""
    return "; ".join(str(item["msg"]) for item in error.errors(include_input=False))


def _service_error_message(error: Exception, fallback: str) -> str:
    """Translate common Gemini API status codes without exposing request details."""
    cause: BaseException | None = error
    while cause is not None:
        code = getattr(cause, "code", None)
        if code == 503:
            return "Gemini is temporarily busy (503). Try again shortly or configure another Flash model."
        if code == 429:
            return "Gemini quota or rate limit reached (429). Try again later or check your quota."
        cause = cause.__cause__
    return fallback


def _debug_panel() -> None:
    with st.expander("Debug"):
        active_index = st.session_state.active_index
        search_queries = getattr(active_index, "last_queries", ())
        st.subheader("Search queries")
        if search_queries:
            for number, search_query in enumerate(search_queries, start=1):
                st.write(f"{number}. {search_query}")
        else:
            st.caption("None yet.")
        parsed_rows = [
            {
                "source": element.source_name,
                "page": element.page_number,
                "section": element.section_title,
                "element_type": element.category,
                "text": element.text,
                "table_headers": ", ".join(element.table_headers),
            }
            for document in st.session_state.parsed_documents
            for element in document.elements
        ]
        chunk_rows = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source_name,
                "page": chunk.page_number,
                "section": chunk.heading,
                "element_type": chunk.element_type,
                "characters": len(chunk.text),
                "text": chunk.text,
            }
            for chunk in st.session_state.chunks
        ]
        retrieved_rows = [
            {
                "rank": result.rank,
                "chunk_id": result.chunk_id,
                "distance": result.distance,
                "bm25_score": result.lexical_score,
                "hybrid_score": result.retrieval_score,
                "rerank_score": result.rerank_score,
                "source": result.source,
                "page": result.page,
                "section": result.section,
                "element_type": result.element_type,
                "text": result.text,
            }
            for result in st.session_state.last_retrieved
        ]
        for title, rows in (
            ("Parsed elements", parsed_rows),
            ("Chunks", chunk_rows),
            ("Last retrieved chunks and scores", retrieved_rows),
        ):
            st.subheader(title)
            if rows:
                st.dataframe(rows, hide_index=True)
            else:
                st.caption("None yet.")
        st.caption("Cosine distance: lower is closer. BM25, hybrid, and rerank scores: higher ranks first.")


def main() -> None:
    st.set_page_config(page_title="Document RAG", page_icon="📄")
    _initialize_state()
    st.title("Document RAG")
    st.caption("Upload documents, process them, and ask questions grounded in their contents.")

    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader(
            "Upload PDF, TXT, Markdown, DOC, DOCX, or CSV files",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            key="uploads",
        ) or []
        try:
            settings = load_config()
        except ValidationError as exc:
            st.error("Configuration error: " + _validation_message(exc))
            return
        url_warning = unstructured_api_url_warning(os.getenv("UNSTRUCTURED_API_URL"))
        if url_warning:
            st.warning(url_warning)
        hosted_parsing = (
            settings.unstructured_api_key is not None
            and settings.unstructured_api_url is not None
        )
        st.caption(
            "Parsing: Unstructured API (local fallback if key is rejected)"
            if hosted_parsing
            else "Parsing: local Unstructured"
        )
        if settings.unstructured_api_key is not None and not hosted_parsing:
            st.caption("Set UNSTRUCTURED_API_URL to enable hosted parsing.")
        if files:
            st.write(f"Selected files ({len(files)})")
            for file in files:
                st.write(f"• {file.name}")
        else:
            st.caption("No files selected.")

        signature = _selection_signature(files)
        if st.button("Process Documents", disabled=not files, type="primary", use_container_width=True):
            try:
                with st.spinner("Parsing and indexing documents..."):
                    _build(files, signature)
            except ValidationError as exc:
                st.error(_validation_message(exc))
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(_service_error_message(exc, "Could not build the index. Check document parsing and Gemini access."))
            else:
                st.success(f"Processed {len(st.session_state.parsed_documents)} documents into {st.session_state.active_index.chunk_count} chunks.")

        if st.button("Reset", disabled=st.session_state.active_index is None):
            try:
                reset_index(st.session_state.active_index)
            except Exception:
                st.error("Could not reset the index.")
            else:
                _clear_active_state()
                st.success("Index reset. Process documents to ask questions.")

        ready = is_index_ready(st.session_state.active_index)
        selected_is_active = ready and signature == st.session_state.active_signature
        if ready:
            st.success("Index ready for questions")
            if not selected_is_active:
                st.warning("Selected files differ from the active index. Process Documents to replace it.")
        else:
            st.info("No index ready")
        st.metric("Documents", len(st.session_state.parsed_documents) if ready else 0)
        st.metric("Chunks", st.session_state.active_index.chunk_count if ready else 0)
        if ready:
            for document in st.session_state.parsed_documents:
                for warning in document.warnings:
                    st.warning(warning)
        if ready:
            st.caption("Indexed documents: " + ", ".join(st.session_state.active_filenames))

    st.header("Ask a question")
    if ready:
        st.caption(
            f"Searching {len(st.session_state.active_filenames)} indexed "
            f"document{'s' if len(st.session_state.active_filenames) != 1 else ''}."
        )
    question = st.text_area(
        "Question",
        placeholder="What do the indexed documents say?",
        key="question",
        height=120,
        disabled=not ready,
    )
    ask_submitted = st.button("Ask Question", disabled=not ready, type="primary", use_container_width=True)
    if ask_submitted:
        if not question.strip():
            st.warning("Enter a question before asking.")
        else:
            try:
                with st.spinner("Finding evidence and generating an answer..."):
                    _ask(question)
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(_service_error_message(exc, "Could not answer the question. Check Gemini access and try again."))

    if not ready:
        st.info("Process documents to ask questions.")
    elif not selected_is_active:
        st.info("Questions use the indexed documents shown in the sidebar until you build a replacement index.")

    answer = (
        st.session_state.last_answer
        if ready and question == st.session_state.last_question
        else None
    )
    query_warning = getattr(st.session_state.active_index, "query_warning", None)
    if ready and st.session_state.last_retrieved and query_warning:
        st.warning(query_warning)
    rerank_warning = getattr(st.session_state.active_index, "rerank_warning", None)
    if ready and st.session_state.last_retrieved and rerank_warning:
        st.warning(rerank_warning)
    if answer is not None:
        st.subheader("Answer")
        st.write(answer.answer)
        st.subheader("Citations")
        if answer.citations:
            for citation in answer.citations:
                location = [citation.source]
                if citation.page is not None:
                    location.append(f"page {citation.page}")
                if citation.section:
                    location.append(citation.section)
                st.write(f"[{citation.evidence_id}] " + " · ".join(location))
        else:
            st.caption("No supporting citation was available.")

        st.subheader("Supporting passages")
        for result in st.session_state.last_retrieved:
            location = [result.source]
            if result.page is not None:
                location.append(f"page {result.page}")
            if result.section:
                location.append(result.section)
            with st.expander(f"{result.rank}. " + " · ".join(location)):
                st.write(result.text)
                if result.distance is not None:
                    st.caption(f"Cosine distance: {result.distance:.4f}")
                if result.rerank_score is not None:
                    st.caption(f"Rerank score: {result.rerank_score:.4f}")

    _debug_panel()
    st.divider()
    st.caption("Developed by [savindu.dev](https://savindu-dev.web.app/)")
