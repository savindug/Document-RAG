"""Section-aware chunking with bounded prose and intact table rows."""

from collections.abc import Iterable
from hashlib import sha256
from html.parser import HTMLParser
import textwrap

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import ElementMetadata, Text

from document_rag.config import AppConfig
from document_rag.models import DocumentChunk, ParsedDocument, ParsedElement


class ChunkingError(ValueError):
    """The document cannot be chunked within the configured limits."""


class _TableRows(HTMLParser):
    """Extract ordered rows from Unstructured table HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None


def _table_lines(element: ParsedElement) -> list[str]:
    if element.table_html:
        parser = _TableRows()
        parser.feed(element.table_html)
        parser.close()
        rows = parser.rows
        if rows and element.table_headers and rows[0] == element.table_headers:
            rows = rows[1:]
        if rows:
            return [" | ".join(row) for row in rows]
    return [line.strip() for line in element.text.splitlines() if line.strip()]


def _split_bounded(text: str, limit: int) -> list[str]:
    """Split an indivisible paragraph or table row without exceeding the hard limit."""
    return [
        part.strip()
        for part in textwrap.wrap(
            text,
            width=limit,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        if part.strip()
    ]


def _rebalance_short_tail(parts: list[str], limit: int, minimum: int) -> list[str]:
    """Avoid a tiny final prose chunk when the previous chunk can share content."""
    if len(parts) < 2 or len(parts[-1]) >= minimum:
        return parts
    combined = f"{parts[-2]}\n\n{parts[-1]}"
    if len(combined) <= limit:
        return [*parts[:-2], combined]

    midpoint = len(combined) // 2
    lower = max(minimum, len(combined) - limit)
    upper = min(limit, len(combined) - minimum)
    split_at = combined.rfind(" ", lower, min(upper + 1, midpoint + 1))
    if split_at < lower:
        split_at = combined.find(" ", midpoint, upper + 1)
    if split_at < lower or split_at > upper:
        split_at = midpoint
    return [*parts[:-2], combined[:split_at].strip(), combined[split_at:].strip()]


def _prose_parts(elements: list[ParsedElement], heading: str | None, settings: AppConfig) -> list[str]:
    prefix_len = len(heading) + 2 if heading else 0
    budget = settings.chunk_max_chars - prefix_len
    if budget < 50:
        raise ChunkingError("Section title leaves too little room for chunk text.")

    unstructured_elements = [
        Text(element.text, metadata=ElementMetadata(page_number=element.page_number))
        for element in elements
    ]
    raw_chunks = chunk_by_title(
        unstructured_elements,
        max_characters=budget,
        new_after_n_chars=min(budget, max(50, settings.chunk_target_chars - prefix_len)),
        combine_text_under_n_chars=min(settings.chunk_min_chars, budget),
        include_orig_elements=False,
        overlap=0,
    )
    parts: list[str] = []
    for raw in raw_chunks:
        text = raw.text.strip()
        if text:
            parts.extend(_split_bounded(text, budget) if len(text) > budget else [text])
    return _rebalance_short_tail(parts, budget, settings.chunk_min_chars)


def _table_parts(element: ParsedElement, heading: str | None, settings: AppConfig) -> list[str]:
    prefix = f"{heading}\n\n" if heading else ""
    header = " | ".join(element.table_headers)
    header_prefix = f"{header}\n" if header else ""
    budget = settings.chunk_max_chars - len(prefix) - len(header_prefix)
    if budget < 50:
        raise ChunkingError("Table heading or headers leave too little room for table rows.")

    lines = _table_lines(element)
    parts: list[str] = []
    current: list[str] = []
    for line in lines:
        for row_part in _split_bounded(line, budget):
            candidate = "\n".join([*current, row_part])
            if current and len(candidate) > budget:
                parts.append("\n".join(current))
                current = [row_part]
            else:
                current.append(row_part)
    if current:
        parts.append("\n".join(current))
    return [f"{prefix}{header_prefix}{part}" for part in parts if part.strip()]


def _make_chunk(
    document: ParsedDocument,
    text: str,
    heading: str | None,
    page_number: int | None,
    element_type: str,
    index: int,
) -> DocumentChunk:
    payload = f"{document.source_id}\0{index}\0{text}".encode("utf-8")
    chunk_id = f"{document.source_id}:{sha256(payload).hexdigest()[:16]}"
    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=document.source_id,
        source_name=document.source_name,
        text=text,
        embedding_text=text,
        page_number=page_number,
        heading=heading,
        element_type=element_type,
    )


def chunk_documents(
    documents: Iterable[ParsedDocument], config: AppConfig | None = None
) -> list[DocumentChunk]:
    """Chunk each source independently, retaining page and section attribution."""
    settings = config or AppConfig()
    chunks: list[DocumentChunk] = []

    for document in documents:
        starting_chunk_count = len(chunks)
        group: list[ParsedElement] = []
        group_heading: str | None = None
        group_page: int | None = None
        document_chunk_index = 0

        def append_chunk(
            text: str, heading: str | None, page_number: int | None, element_type: str
        ) -> None:
            nonlocal document_chunk_index
            chunks.append(
                _make_chunk(document, text, heading, page_number, element_type, document_chunk_index)
            )
            document_chunk_index += 1

        def flush_group() -> None:
            nonlocal group
            if not group:
                return
            types = {element.category for element in group}
            element_type = next(iter(types)) if len(types) == 1 else "CompositeElement"
            for body in _prose_parts(group, group_heading, settings):
                text = f"{group_heading}\n\n{body}" if group_heading else body
                append_chunk(text, group_heading, group_page, element_type)
            group = []

        current_heading: str | None = None
        for element in document.elements:
            if element.category == "Title":
                flush_group()
                current_heading = element.text.strip()
                continue

            heading = element.section_title or current_heading
            if element.category == "Table":
                flush_group()
                for text in _table_parts(element, heading, settings):
                    append_chunk(text, heading, element.page_number, "Table")
                continue

            if (heading, element.page_number) != (group_heading, group_page) and group:
                flush_group()
            group_heading, group_page = heading, element.page_number
            if element.text.strip():
                group.append(element)

        flush_group()
        if len(chunks) == starting_chunk_count:
            raise ChunkingError(f"No meaningful content to chunk in {document.source_name}.")

    if len(chunks) > settings.max_chunks:
        raise ChunkingError(f"Document set exceeds the {settings.max_chunks} chunk limit.")
    if any(not chunk.text.strip() or len(chunk.text) > settings.chunk_max_chars for chunk in chunks):
        raise ChunkingError("A chunk is blank or exceeds the configured size limit.")
    return chunks
