"""Document and answer shape checks."""

import pytest
from pydantic import ValidationError

from document_rag.models import AnswerCitation, GroundedAnswer, UploadedDocument


def test_upload_rejects_unsupported_file() -> None:
    with pytest.raises(ValidationError, match="Only PDF, TXT, Markdown, DOC, DOCX, and CSV"):
        UploadedDocument(filename="photo.png", content=b"image")
    with pytest.raises(ValidationError, match="Only PDF, TXT, Markdown, DOC, DOCX, and CSV"):
        UploadedDocument(filename="pdf", content=b"data")


def test_grounded_answer_requires_citation() -> None:
    with pytest.raises(ValidationError, match="cite at least one"):
        GroundedAnswer(answer="A fact")
    answer = GroundedAnswer(
        answer="A fact [S1]",
        evidence_ids=["S1"],
        citations=[
            AnswerCitation(
                evidence_id="S1", chunk_id="chunk-1", source="notes.txt", text="A fact"
            )
        ],
    )
    assert answer.citations[0].chunk_id == "chunk-1"
