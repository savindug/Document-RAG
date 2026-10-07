"""Validate uploaded files and preserve Unstructured's per-element output."""

from collections.abc import Iterable
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path, PurePath
from shutil import which
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from document_rag.config import AppConfig
from document_rag.models import SUPPORTED_SUFFIXES, ParsedDocument, ParsedElement, UploadedDocument


CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".csv": "text/csv",
}
TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".csv"})


class UploadValidationError(ValueError):
    """The uploaded set cannot be parsed safely."""


class DocumentParsingError(ValueError):
    """A supported file could not be turned into usable elements."""


class _TableHeaderParser(HTMLParser):
    """Read semantic table headers, including cells inside a thead."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headers: list[str] = []
        self.first_row: list[str] = []
        self._thead_depth = 0
        self._row_count = 0
        self._in_first_row = False
        self._cell_parts: list[str] | None = None
        self._cell_is_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "thead":
            self._thead_depth += 1
        elif tag == "tr":
            self._in_first_row = self._row_count == 0
            self._row_count += 1
        elif tag in {"th", "td"}:
            self._cell_parts = []
            self._cell_is_header = tag == "th" or self._thead_depth > 0

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell_parts is not None:
            cell_text = " ".join("".join(self._cell_parts).split())
            if cell_text:
                if self._cell_is_header:
                    self.headers.append(cell_text)
                if self._in_first_row:
                    self.first_row.append(cell_text)
            self._cell_parts = None
        elif tag == "tr":
            self._in_first_row = False
        elif tag == "thead":
            self._thead_depth = max(0, self._thead_depth - 1)


def validate_uploads(
    uploads: Iterable[UploadedDocument], config: AppConfig | None = None
) -> list[UploadedDocument]:
    """Validate a complete upload set before parsing any file."""
    settings = config or AppConfig()
    files = list(uploads)
    if not files:
        raise UploadValidationError("Upload at least one PDF, TXT, Markdown, DOC, DOCX, or CSV file.")
    if len(files) > settings.max_files:
        raise UploadValidationError(f"Upload at most {settings.max_files} files at a time.")

    seen_names: set[str] = set()
    for upload in files:
        if not isinstance(upload, UploadedDocument):
            raise UploadValidationError("Each upload must be an UploadedDocument.")
        suffix = PurePath(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise UploadValidationError(f"Unsupported file: {upload.filename}.")
        if len(upload.content) > settings.max_file_bytes:
            raise UploadValidationError(
                f"{upload.filename} exceeds the {settings.max_file_bytes} byte file limit."
            )
        if suffix not in TEXT_SUFFIXES and not upload.content.strip():
            raise UploadValidationError(f"{upload.filename} is empty.")
        if suffix in TEXT_SUFFIXES:
            try:
                text = upload.content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise UploadValidationError(f"{upload.filename} must be UTF-8 text.") from exc
            if not text.strip():
                raise UploadValidationError(f"{upload.filename} contains no text.")
        name_key = upload.filename.casefold()
        if name_key in seen_names:
            raise UploadValidationError(f"Duplicate filename: {upload.filename}.")
        seen_names.add(name_key)
    return files


def _partition_upload(upload: UploadedDocument, *, pdf_strategy: str = "hi_res") -> list[Any]:
    """Dispatch to the local Unstructured library without writing the upload to disk."""
    from unstructured.partition.auto import partition

    suffix = PurePath(upload.filename).suffix.lower()
    kwargs: dict[str, Any] = {
        "file": BytesIO(upload.content),
        "content_type": CONTENT_TYPES[suffix],
        "metadata_filename": upload.filename,
    }
    if suffix == ".pdf":
        # OCR and table HTML require hi_res; fast can read embedded PDF text without OCR.
        kwargs.update(
            strategy=pdf_strategy,
            pdf_infer_table_structure=pdf_strategy == "hi_res",
        )
    elif suffix == ".csv":
        # Unstructured drops the first row from table text unless include_header is set.
        kwargs.update(include_header=True, infer_table_structure=True, encoding="utf-8-sig")
    elif suffix in {".doc", ".docx"}:
        kwargs["infer_table_structure"] = True
    return partition(**kwargs)


def _partition_via_api(upload: UploadedDocument, api_key: str, api_url: str) -> list[Any]:
    """Use LangChain's loader for hosted element-level parsing."""
    from langchain_unstructured import UnstructuredLoader
    from unstructured_client import UnstructuredClient
    from unstructured_client.utils import BackoffStrategy, RetryConfig

    suffix = PurePath(upload.filename).suffix.lower()
    server_url = api_url.rstrip("/").removesuffix("/general/v0/general")
    client = UnstructuredClient(
        api_key_auth=api_key,
        server_url=server_url,
        timeout_ms=120_000,
        retry_config=RetryConfig(
            strategy="backoff",
            backoff=BackoffStrategy(
                initial_interval=500,
                max_interval=2_000,
                exponent=2,
                max_elapsed_time=10_000,
            ),
            retry_connection_errors=False,
        ),
    )
    # The loader derives the API filename from file_path. A temporary file keeps
    # the extension available for format detection and is removed after loading.
    with TemporaryDirectory(prefix="document-rag-") as directory:
        path = Path(directory) / PurePath(upload.filename).name
        path.write_bytes(upload.content)
        loader = UnstructuredLoader(
            file_path=str(path),
            partition_via_api=True,
            client=client,
            strategy="hi_res" if suffix == ".pdf" else "auto",
            pdf_infer_table_structure=suffix == ".pdf",
            split_pdf_page=False,
        )
        documents = loader.load()
    return [
        SimpleNamespace(
            text=document.page_content,
            category=document.metadata.get("category") or "UncategorizedText",
            metadata=SimpleNamespace(**document.metadata),
        )
        for document in documents
    ]


def _missing_tesseract(error: BaseException) -> bool:
    """Recognize either pytesseract wrapper's missing-executable error."""
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        if type(current).__name__ == "TesseractNotFoundError":
            return True
        seen.add(id(current))
        pending.extend(
            nested for nested in (current.__cause__, current.__context__) if nested is not None
        )
    return False


def _partition_locally(upload: UploadedDocument) -> tuple[list[Any], bool]:
    """Parse locally, retrying text-based PDFs without OCR when necessary."""
    try:
        return _partition_upload(upload), False
    except Exception as exc:
        if upload.filename.lower().endswith(".pdf") and _missing_tesseract(exc):
            try:
                return _partition_upload(upload, pdf_strategy="fast"), True
            except Exception as fast_exc:
                raise DocumentParsingError(
                    f"Could not extract text from {upload.filename} without OCR. "
                    "Install Tesseract OCR and add tesseract.exe to PATH for scanned PDFs."
                ) from fast_exc
        if upload.filename.lower().endswith(".doc") and which("soffice") is None:
            raise DocumentParsingError(
                "Parsing legacy DOC files requires LibreOffice (soffice) on PATH."
            ) from exc
        raise DocumentParsingError(
            f"Could not parse {upload.filename} ({type(exc).__name__})."
        ) from exc


def _table_headers(html: str | None, *, first_row_is_header: bool = False) -> list[str]:
    if not html:
        return []
    parser = _TableHeaderParser()
    parser.feed(html)
    parser.close()
    return parser.headers or (parser.first_row if first_row_is_header else [])


def parse_documents(
    uploads: Iterable[UploadedDocument], config: AppConfig | None = None
) -> list[ParsedDocument]:
    """Return ordered elements grouped by source, with no cross-file text merging."""
    files = validate_uploads(uploads, config)
    parsed: list[ParsedDocument] = []
    api_url = config.unstructured_api_url if config is not None else None
    api_key = (
        config.unstructured_api_key.get_secret_value()
        if config is not None and config.unstructured_api_key is not None and api_url
        else None
    )
    hosted_auth_failed = False

    for upload in files:
        source_id = sha256(upload.filename.encode("utf-8") + b"\0" + upload.content).hexdigest()
        used_fast_fallback = False
        used_auth_fallback = hosted_auth_failed
        if api_key is not None and not hosted_auth_failed:
            try:
                raw_elements = _partition_via_api(upload, api_key, api_url)
            except Exception as exc:
                from httpx import ConnectError, TimeoutException

                if isinstance(exc, ConnectError):
                    raise DocumentParsingError(
                        "Could not connect to the Unstructured Partition API. "
                        "Check UNSTRUCTURED_API_URL, DNS, proxy, and TLS settings. "
                        "The standard Partition host is api.unstructuredapp.io. "
                        "This connection failure does not establish whether the API key is valid."
                    ) from exc
                if isinstance(exc, TimeoutException):
                    raise DocumentParsingError(
                        "The Unstructured Partition API request timed out. Try again or upload a smaller file."
                    ) from exc
                if getattr(exc, "status_code", None) == 401:
                    hosted_auth_failed = True
                    used_auth_fallback = True
                    try:
                        raw_elements, used_fast_fallback = _partition_locally(upload)
                    except DocumentParsingError as local_exc:
                        raise DocumentParsingError(
                            "Unstructured Partition API rejected the configured key (401). "
                            f"Local parsing also failed: {local_exc}"
                        ) from local_exc
                else:
                    raise DocumentParsingError(
                        f"Unstructured API could not parse {upload.filename} ({type(exc).__name__}). "
                        "Check UNSTRUCTURED_API_KEY, API access, and service availability."
                    ) from exc
        else:
            raw_elements, used_fast_fallback = _partition_locally(upload)

        elements: list[ParsedElement] = []
        current_section: str | None = None
        for raw in raw_elements:
            text = str(getattr(raw, "text", "") or "").strip()
            if not text:
                continue
            category = str(getattr(raw, "category", type(raw).__name__))
            if category == "Title":
                current_section = text
            metadata = getattr(raw, "metadata", None)
            table_html = getattr(metadata, "text_as_html", None) if category == "Table" else None
            elements.append(
                ParsedElement(
                    element_id=f"{source_id}:{len(elements)}",
                    source_id=source_id,
                    source_name=upload.filename,
                    text=text,
                    category=category,
                    page_number=getattr(metadata, "page_number", None),
                    section_title=current_section,
                    table_html=table_html,
                    table_headers=_table_headers(
                        table_html, first_row_is_header=upload.filename.lower().endswith(".csv")
                    ),
                )
            )
        if not elements:
            if used_fast_fallback:
                raise DocumentParsingError(
                    f"No extractable text found in {upload.filename}. "
                    "Install Tesseract OCR and add tesseract.exe to PATH for scanned PDFs."
                )
            raise DocumentParsingError(f"No extractable text found in {upload.filename}.")
        warnings = (
            [
                f"{upload.filename} was parsed without OCR because Tesseract is unavailable. "
                "PDF table structure and scanned text may be incomplete."
            ]
            if used_fast_fallback
            else []
        )
        if used_auth_fallback:
            warnings.insert(
                0,
                "Unstructured Partition API rejected the configured key (401); "
                f"{upload.filename} was parsed locally. Configure a valid Partition API key "
                "for hosted parsing.",
            )
        parsed.append(
            ParsedDocument(
                source_id=source_id,
                source_name=upload.filename,
                elements=elements,
                warnings=warnings,
            )
        )

    return parsed
