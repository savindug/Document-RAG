"""Structure, size, source attribution, and deterministic chunk IDs."""

import pytest

from document_rag.chunking import ChunkingError, chunk_documents
from document_rag.config import AppConfig
from document_rag.models import ParsedDocument, ParsedElement


def element(
    index: int,
    text: str,
    category: str = "NarrativeText",
    *,
    source_id: str = "source-a",
    source_name: str = "report.md",
    page: int | None = 1,
    section: str | None = None,
    table_html: str | None = None,
    headers: list[str] | None = None,
) -> ParsedElement:
    return ParsedElement(
        element_id=f"{source_id}:{index}",
        source_id=source_id,
        source_name=source_name,
        text=text,
        category=category,
        page_number=page,
        section_title=section,
        table_html=table_html,
        table_headers=headers or [],
    )


def document(*elements: ParsedElement, source_id: str = "source-a", source_name: str = "report.md") -> ParsedDocument:
    return ParsedDocument(source_id=source_id, source_name=source_name, elements=list(elements))


def test_headings_are_in_text_and_sections_do_not_mix() -> None:
    parsed = document(
        element(0, "Overview", "Title", section="Overview"),
        element(1, "The project began in 2024.", section="Overview"),
        element(2, "A short related detail.", section="Overview"),
        element(3, "Results", "Title", page=2, section="Results"),
        element(4, "Revenue grew by twelve percent.", page=2, section="Results"),
    )

    chunks = chunk_documents([parsed])

    assert len(chunks) == 2
    assert chunks[0].text.startswith("Overview\n\n")
    assert "The project began" in chunks[0].text
    assert "Results" not in chunks[0].text
    assert chunks[1].text.startswith("Results\n\n")
    assert chunks[1].page_number == 2
    assert all(chunk.source_name == "report.md" for chunk in chunks)
    assert all(chunk.source_id == "source-a" for chunk in chunks)
    assert all(chunk.element_type == "NarrativeText" for chunk in chunks)


def test_table_headers_repeat_when_rows_are_split() -> None:
    rows = "".join(f"<tr><td>Item {i}</td><td>{i * 10}</td></tr>" for i in range(20))
    html = f"<table><tr><td>Item</td><td>Amount</td></tr>{rows}</table>"
    parsed = document(
        element(0, "Sales", "Title", section="Sales"),
        element(
            1,
            "Item Amount " + " ".join(f"Item {i} {i * 10}" for i in range(20)),
            "Table",
            section="Sales",
            table_html=html,
            headers=["Item", "Amount"],
        ),
    )
    settings = AppConfig(chunk_max_chars=140, chunk_target_chars=110, chunk_min_chars=25)

    chunks = chunk_documents([parsed], settings)

    assert len(chunks) > 1
    assert all(chunk.element_type == "Table" for chunk in chunks)
    assert all(chunk.text.startswith("Sales\n\nItem | Amount\n") for chunk in chunks)
    assert all(len(chunk.text) <= settings.chunk_max_chars for chunk in chunks)
    assert sum("Item 19 | 190" in chunk.text for chunk in chunks) == 1
    assert sum("Item 0 | 0" in chunk.text for chunk in chunks) == 1


def test_long_prose_is_bounded_and_has_no_title_only_or_blank_chunks() -> None:
    settings = AppConfig(chunk_max_chars=220, chunk_target_chars=170, chunk_min_chars=35)
    body = "A meaningful sentence with a useful fact. " * 45
    parsed = document(
        element(0, "Important section", "Title", section="Important section"),
        element(1, body, section="Important section"),
    )

    chunks = chunk_documents([parsed], settings)

    assert len(chunks) > 2
    assert all(chunk.text.startswith("Important section\n\n") for chunk in chunks)
    assert all(len(chunk.text) <= settings.chunk_max_chars for chunk in chunks)
    assert all(len(chunk.text.split("\n\n", 1)[1]) >= settings.chunk_min_chars for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_ids_are_stable_and_independent_of_other_documents() -> None:
    first = document(element(0, "A stable fact in the document."))
    second = document(
        element(0, "Another fact.", source_id="source-b", source_name="other.txt"),
        source_id="source-b",
        source_name="other.txt",
    )

    first_ids = [chunk.chunk_id for chunk in chunk_documents([first])]
    repeated_ids = [chunk.chunk_id for chunk in chunk_documents([first])]
    reordered_ids = [chunk.chunk_id for chunk in chunk_documents([second, first]) if chunk.source_id == "source-a"]

    assert first_ids == repeated_ids == reordered_ids
    assert all(chunk_id.startswith("source-a:") for chunk_id in first_ids)


def test_chunk_limit_rejects_oversized_dataset() -> None:
    parsed = document(element(0, "A fact. " * 100))
    settings = AppConfig(max_chunks=1, chunk_max_chars=150, chunk_target_chars=100)

    with pytest.raises(ChunkingError, match="chunk limit"):
        chunk_documents([parsed], settings)


def test_title_only_document_has_no_meaningless_chunk() -> None:
    parsed = document(element(0, "Just a heading", "Title", section="Just a heading"))

    with pytest.raises(ChunkingError, match="No meaningful content"):
        chunk_documents([parsed])


def test_oversized_table_row_stays_within_limit() -> None:
    html = f"<table><tr><th>Label</th></tr><tr><td>{'word ' * 120}</td></tr></table>"
    parsed = document(
        element(
            0,
            "Label " + "word " * 120,
            "Table",
            table_html=html,
            headers=["Label"],
        )
    )
    settings = AppConfig(chunk_max_chars=130, chunk_target_chars=100, chunk_min_chars=25)

    chunks = chunk_documents([parsed], settings)

    assert len(chunks) > 1
    assert all(chunk.text.startswith("Label\n") for chunk in chunks)
    assert all(len(chunk.text) <= settings.chunk_max_chars for chunk in chunks)
