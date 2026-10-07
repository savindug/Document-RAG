"""Typed contracts shared across ingestion, retrieval, and answer generation."""

from pathlib import PurePath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SUPPORTED_SUFFIXES = frozenset({".pdf", ".txt", ".md", ".markdown", ".doc", ".docx", ".csv"})


class UploadedDocument(BaseModel):
    """A file received from the Streamlit uploader."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(min_length=1)
    content: bytes = Field(min_length=1, repr=False)

    @model_validator(mode="after")
    def validate_type(self) -> "UploadedDocument":
        if PurePath(self.filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("Only PDF, TXT, Markdown, DOC, DOCX, and CSV files are supported.")
        return self


class ParsedElement(BaseModel):
    """Text extracted from an uploaded document, with source attribution."""

    model_config = ConfigDict(frozen=True)

    element_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    text: str = Field(min_length=1)
    category: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    section_title: str | None = None
    table_html: str | None = None
    table_headers: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """One uploaded source and its ordered, uncombined elements."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    elements: list[ParsedElement] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """A source passage and the text used to embed it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    text: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    heading: str | None = None
    element_type: str | None = None


class RetrievedChunk(BaseModel):
    """A ranked chunk returned by the active dataset's retriever."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    rank: int = Field(ge=1)
    distance: float | None = Field(default=None, ge=0)


class RetrievalResult(BaseModel):
    """A flat, source-attributed search hit for the RAG retrieval boundary."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    element_type: str | None = None
    distance: float | None = Field(default=None, ge=0)
    lexical_score: float | None = Field(default=None, ge=0)
    retrieval_score: float | None = Field(default=None, ge=0)
    rerank_score: float | None = None


class QueryPlan(BaseModel):
    """Bounded search questions derived from a multipart user question."""

    model_config = ConfigDict(extra="forbid")

    subqueries: list[str] = Field(default_factory=list)


class AnswerCitation(BaseModel):
    """A reference to one of the chunks supplied as evidence."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(min_length=2)
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    text: str = Field(min_length=1)


class AnswerDraft(BaseModel):
    """The exact structured response requested from Gemini."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        description="Answer using only supplied evidence; cite IDs inline like [S1].",
    )
    evidence_ids: list[str] = Field(
        description="Unique IDs supporting the answer, such as S1 and S2; empty when evidence is insufficient."
    )
    insufficient_evidence: bool = Field(
        description="True when the supplied evidence cannot answer the question."
    )

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Answer must not be blank.")
        return value


class GroundedAnswer(BaseModel):
    """The validated answer shape returned to the UI."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def validate_citations(self) -> "GroundedAnswer":
        if self.insufficient_evidence:
            if self.citations or self.evidence_ids:
                raise ValueError("An insufficient-evidence answer cannot contain citations.")
        elif not self.citations or [item.evidence_id for item in self.citations] != self.evidence_ids:
            raise ValueError("A grounded answer must cite at least one retrieved chunk.")
        return self
