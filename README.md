# Document RAG

A Streamlit app that accepts documents at runtime, builds an in-memory index, and answers questions from uploaded evidence. Process a new document set to replace the active index without changing code.

## Features

- Upload multiple PDF, TXT, Markdown (`.md`, `.markdown`), DOC, DOCX, or CSV files. Validation rejects empty, unsupported, oversized, or excessive uploads.
- Parse with Unstructured, retaining filename, page, section/title, element type, and table context where available.
- Split prose with Unstructured `chunk_by_title`; include headings in chunk text and retain table headers with rows. Chunk IDs are deterministic.
- Embed with Gemini `gemini-embedding-2` and store text and metadata in an in-memory Chroma collection with cosine distance.
- Search with Chroma semantic retrieval and in-memory BM25, combine candidates with reciprocal rank fusion, then rerank with a local cross encoder. Select `TOP_K` after reranking.
- Optionally decompose comparison and multi-part questions into focused searches; also search the original question.
- Generate grounded Gemini Flash Lite answers with `[S1]` evidence citations, source metadata, supporting passages, and an insufficient-evidence response when needed.
- Inspect parsed elements, chunks, search queries, retrieval scores, and metadata in the Streamlit Debug section.
- Replace or reset the active document set. Each successful build uses a fresh Chroma collection, preventing old/new mixing.

## Technology decisions

**LangChain** connects Gemini embeddings, the answer model, and Chroma through standard components. Its document and model interfaces let filename, page, and section metadata travel through indexing and retrieval for citations. LangChain handles much of the model/vector-store wiring; explicit prompt messages and separate pipeline modules make the flow inspectable and let us compare chunking or retrieval settings without rewriting the app.

**Unstructured** extracts document elements and their structure across formats. `chunk_by_title` groups prose under section headings so meaning survives splitting. Table handling keeps headers with rows. This makes new runtime document sets usable without parser or chunking code changes.

**Hybrid retrieval helps find the right evidence; a cross encoder helps put the best evidence first.** BM25 finds exact terms, Chroma finds semantic matches, reciprocal rank fusion combines candidates, and the cross encoder scores them against the original question. `TOP_K` selects the best-ranked available passages after reranking. Scores are ranking signals, not answer-confidence estimates.

**Query decomposition** helps complex questions with comparisons or multiple tasks. A planner can produce up to `MAX_SUBQUERIES` focused queries; retrieval always includes the original question. If planning fails, retrieval continues with the original question.

## Install

Python 3.12 is recommended. From the project root, on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS/Linux, create and activate the environment with `python3 -m venv .venv` and `source .venv/bin/activate`, then use the same pip commands and `cp .env.example .env`.

Scanned PDFs require Tesseract OCR; legacy `.doc` files may require LibreOffice. The first reranking run may download model weights from Hugging Face. Hosted Unstructured Partition is optional.

## Configure `.env`

Set `GOOGLE_API_KEY` in `.env`. Do not commit this file. Environment variables override `.env`; restart Streamlit after editing settings.

| Variable                      | Default                               | Purpose                                                                                             |
| ----------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `GOOGLE_API_KEY`              | Required                              | Gemini embeddings, planning, and answers.                                                           |
| `GEMINI_MODEL`                | `gemini-3.8-flash`                    | Model used for planning and answers; the key must have access.                                      |
| `GEMINI_EMBEDDING_MODEL`      | `gemini-embedding-2`                  | Embedding model; requested output is 768 dimensions.                                                |
| `TOP_K`                       | `5`                                   | Maximum final passages after reranking.                                                             |
| `RETRIEVAL_CANDIDATES`        | `20`                                  | Candidate-pool target before final selection; actual searches depend on query count and index size. |
| `QUERY_DECOMPOSITION_ENABLED` | `true`                                | Plan subqueries for composite questions.                                                            |
| `MAX_SUBQUERIES`              | `3`                                   | Maximum planned subquestions.                                                                       |
| `RERANK_ENABLED`              | `true`                                | Use local cross encoder reranking.                                                                  |
| `RERANK_MODEL`                | `cross-encoder/ms-marco-MiniLM-L6-v2` | Sentence Transformers cross encoder.                                                                |
| `UNSTRUCTURED_API_KEY`        | Empty                                 | Optional hosted Unstructured Partition key.                                                         |
| `UNSTRUCTURED_API_URL`        | Empty                                 | Matching HTTPS Partition endpoint, required with hosted key.                                        |

For hosted parsing, use the Partition URL supplied with your Unstructured account, for example `https://api.unstructuredapp.io/general/v0/general`. The Transform URL `https://transform.unstructured.io/api/v2` is a different API. With no hosted key/URL, parsing runs locally. A hosted 401 may fall back to local parsing; other connection errors are reported.

`document_rag/config.py` also defines limits: 10 files, 10 MiB per file, 200 chunks, and chunk sizes of 2,000 maximum, 1,400 target, and 80 minimum characters. These `AppConfig` fields are not `.env` variables.

## Run and use

```powershell
python -m streamlit run app.py
```

1. Select one or more supported files in the sidebar and click **Process Documents**.
2. When the index has chunks, enter a question in the text area and click **Ask Question**.
3. Read the answer, citations, and supporting passages. Open **Debug** for parsed elements, chunks, search queries, scores, and metadata.
4. Select different files and click **Process Documents** to replace the set. A failed replacement keeps the previous working index. Click **Reset** to remove it.

Streamlit session state keeps the index across reruns. Chroma uses `EphemeralClient`; documents and vectors are not persisted and disappear when the session or process ends. Downloaded cross encoder model weights may be cached locally.

## Architecture

```text
Uploads -> validation -> Unstructured elements -> title/table-aware chunks
        -> Gemini embeddings -> in-memory Chroma
Question -> optional subqueries -> Chroma + BM25 -> rank fusion
         -> cross-encoder reranking -> TOP_K evidence
         -> Gemini structured answer -> validated citations
```

| Module                                                       | Responsibility                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `app.py`, `document_rag/ui.py`                               | Streamlit entry point, session state, upload/question UI, Debug view.    |
| `document_rag/config.py`, `document_rag/models.py`           | Pydantic settings and pipeline data.                                     |
| `document_rag/parsing.py`, `document_rag/chunking.py`        | Upload validation, Unstructured parsing, heading and table-aware chunks. |
| `document_rag/embeddings.py`, `document_rag/vector_store.py` | Gemini embeddings and isolated in-memory Chroma collections.             |
| `document_rag/lexical.py`, `document_rag/query_planning.py`  | BM25 and optional question decomposition.                                |
| `document_rag/retrieval.py`, `document_rag/reranking.py`     | Candidates, rank fusion, reranking, final selection.                     |
| `document_rag/generation.py`                                 | Evidence-only prompting, Pydantic output, citation validation.           |
| `document_rag/evaluation.py`, `evaluation/`                  | Local evaluation cases and metrics.                                      |
| `tests/`, `scripts/`                                         | Offline tests and optional live smoke checks.                            |

Prompts label user questions and retrieved passages as untrusted input. System instructions tell Gemini to answer only the user's question from supplied evidence. Code maps evidence IDs to actual retrieval metadata and rejects invalid citations. These measures reduce prompt-injection risk but cannot guarantee immunity.

## Test and evaluate

```powershell
python -m pytest -q
```

Tests use fake embeddings and mocked model responses, so they do not call Gemini. Optional live checks require `.env`, network access, and API quota:

```powershell
python scripts/live_pipeline_smoke.py
python scripts/live_ui_smoke.py
```

The small local evaluation measures expected source rank and hit@K, expected keywords, valid citations, and abstention without an LLM judge:

```powershell
python -m document_rag.evaluation --retrieval-only
python -m document_rag.evaluation
```

The offline suite (78 tests) and live pipeline/UI checks passed in the project environment on 2026-10-07. Live results depend on model availability, keys, network, and document content.

## Limitations and next steps

- The index is session-local and temporary; re-upload after restarting. There is no multi-user document store or version tracking.
- OCR and extraction quality depend on Tesseract, document structure, and parser output. Page or section metadata may be missing.
- Initial reranking may need network access for weights. If it is unavailable, retrieval falls back to hybrid ranking with a warning.
- Query decomposition adds a model call for qualifying questions, increasing latency and quota use.
- Citation checks verify IDs refer to retrieved passages; they do not prove every claim is fully supported. Review supporting passages for important decisions.
- Broader document evaluation would help tune chunk sizes, candidate counts, and reranker choice.

See [PLAN.md](PLAN.md) for implementation status and decisions.
