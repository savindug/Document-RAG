# Document RAG implementation plan and status

Status: implemented through 2026-10-07. The app builds a RAG index for the current runtime upload; it does not generate a separate app or retain document versions.

## Goal and implementation decisions

Accept a document set at runtime, index it, answer questions from retrieved evidence, and replace it without code changes. PDF, TXT, Markdown, DOC, DOCX, and CSV are supported.

| Decision | Reason and implemented effect |
| --- | --- |
| LangChain with Gemini and Chroma integrations | Standard embedding, chat, and vector-store components connect the pipeline and keep it testable or replaceable. Filename, page, and section metadata travel with chunks for citations. Prompt messages and structured output provide model wiring; retrieval and generation remain separate modules. |
| Unstructured parsing and `chunk_by_title` | Document elements provide titles, pages, types, and tables. Heading-aware chunks retain context so new document subjects or sets need no parser/chunking code changes. |
| Hybrid retrieval | In-memory BM25 covers exact terms; Gemini vectors in Chroma cover semantic matches. Reciprocal rank fusion gathers candidates from both. |
| Cross encoder reranking | A local cross encoder orders fused candidates against the original question. `TOP_K` is applied after reranking. If the model is unavailable, fused ranking remains usable. |
| Query decomposition | Composite questions can produce focused searches for comparisons or multiple tasks. The original question is always searched, and planning failure falls back to it. |
| Ephemeral index | Each successful set builds a fresh Chroma collection. Replacement and reset prevent old/new mixing. Persistence and version tracking are unnecessary for the current uploaded-set workflow. |

## Implemented flow

1. Streamlit accepts multiple files and validates format, nonempty content, count, and size.
2. Unstructured parses each file into separate elements with source, page, section, element type, and table context where available. Hosted Partition parsing is optional; local parsing is the default.
3. `chunk_by_title` groups prose under headings. Table chunks retain header context. Chunks have deterministic IDs and metadata.
4. `GoogleGenerativeAIEmbeddings` creates 768-dimensional `gemini-embedding-2` vectors. `langchain_chroma.Chroma` stores them in an `EphemeralClient` collection with cosine distance.
5. Gemini Flash Lite may decompose a composite question. Each query searches Chroma and BM25; reciprocal rank fusion combines candidates.
6. A local Sentence Transformers cross encoder reranks candidates against the original question. Retrieval returns top `TOP_K` results with ranks, text, metadata, and available scores.
7. Gemini Flash Lite receives labeled, untrusted evidence `[S1]`, `[S2]`, etc. Pydantic structured output includes the answer, evidence IDs, and insufficient-evidence flag. Code maps IDs back to retrieval metadata and rejects invalid citations.
8. Streamlit displays the answer, citations, supporting passages, and Debug information. Session state preserves the index across reruns; processing a new set replaces it, and Reset removes it.

## Modules

| Area | Module(s) |
| --- | --- |
| Configuration and data contracts | `document_rag/config.py`, `document_rag/models.py` |
| Parsing and chunking | `document_rag/parsing.py`, `document_rag/chunking.py` |
| Embedding and index | `document_rag/embeddings.py`, `document_rag/vector_store.py` |
| Search and planning | `document_rag/lexical.py`, `document_rag/query_planning.py`, `document_rag/retrieval.py`, `document_rag/reranking.py` |
| Answer generation | `document_rag/generation.py` |
| UI and evaluation | `app.py`, `document_rag/ui.py`, `document_rag/evaluation.py`, `evaluation/` |
| Verification | `tests/`, `scripts/live_pipeline_smoke.py`, `scripts/live_ui_smoke.py` |

## Configuration and verification

Runtime variables appear in `.env.example` and are explained in [README.md](README.md). Defaults include `TOP_K=5`, `RETRIEVAL_CANDIDATES=20`, decomposition with at most three subquestions, and cross encoder reranking. File and chunk limits live in `AppConfig`.

The offline pytest suite passed (78 tests) on 2026-10-07. Separate live pipeline and Streamlit smoke checks verified configuration, Gemini embeddings and answers, in-memory Chroma, hybrid retrieval and reranking, query decomposition, citations, replacement, and reset in the project environment. The local evaluation runner checks source rank/hit@K, keywords, citations, and abstention without an LLM judge; its outcomes depend on live models and fixture documents.

## Limitations and possible future work

- Evaluate more real documents, including scanned PDFs and complex tables, then tune chunking and retrieval with measured cases.
- Improve OCR setup guidance or offer hosted OCR when Tesseract is unavailable.
- Add UI progress and latency measurements if large uploads or multi-query searches become common.
- Consider durable storage or multi-user isolation only if scope grows beyond short-lived uploaded sets.

Agents, LangGraph, and automatic document version tracking are outside this implementation.
