"""Small, rule-based RAG evaluation over a fixed local document set."""

import argparse
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_rag.chunking import chunk_documents
from document_rag.config import load_config
from document_rag.generation import generate_answer
from document_rag.models import GroundedAnswer, RetrievalResult, UploadedDocument
from document_rag.parsing import parse_documents
from document_rag.retrieval import retrieve
from document_rag.vector_store import VectorIndex, build_index, reset_index


DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "evaluation" / "cases.json"
_INLINE_ID = re.compile(r"\[(S[1-9]\d*)\]")


class EvaluationCase(BaseModel):
    """Expected evidence for one question; keywords must occur in one passage."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    expected_source: str | None = None
    expected_page: int | None = Field(default=None, ge=1)
    expected_keywords: list[str] = Field(default_factory=list)
    answerable: bool

    @model_validator(mode="after")
    def validate_expected_evidence(self) -> "EvaluationCase":
        if self.answerable and (not self.expected_source or not self.expected_keywords):
            raise ValueError("Answerable cases need an expected source and keywords.")
        if not self.answerable and (self.expected_source or self.expected_page or self.expected_keywords):
            raise ValueError("Unanswerable cases must not declare expected evidence.")
        if any(not word.strip() for word in self.expected_keywords):
            raise ValueError("Expected keywords must not be blank.")
        return self


class EvaluationSuite(BaseModel):
    """Local corpus filenames and their questions."""

    documents: list[str] = Field(min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)


class CaseResult(BaseModel):
    """Measured retrieval and optional answer checks for one case."""

    question: str
    answerable: bool
    expected_source: str | None
    source_in_top_k: bool | None
    source_rank: int | None
    correct_result_rank: int | None
    hit_at_k: bool | None
    valid_citations: bool | None = None
    citations_refer_to_retrieved: bool | None = None
    insufficient_evidence_correct: bool | None = None
    answer_contains_expected_keywords: bool | None = None


class EvaluationReport(BaseModel):
    top_k: int = Field(ge=1)
    cases: list[CaseResult]
    retrieval_hits: int
    retrieval_cases: int
    hit_at_k: float | None
    valid_citation_cases: int | None
    abstention_cases_passed: int | None
    answer_cases: int


def load_suite(path: Path = DEFAULT_FIXTURE) -> tuple[EvaluationSuite, list[UploadedDocument]]:
    """Read a trusted local fixture and its document files."""
    suite = EvaluationSuite.model_validate_json(path.read_text(encoding="utf-8"))
    documents = [
        UploadedDocument(filename=name, content=(path.parent / name).read_bytes())
        for name in suite.documents
    ]
    return suite, documents


def _correct_passage(case: EvaluationCase, result: RetrievalResult) -> bool:
    return (
        result.source == case.expected_source
        and (case.expected_page is None or result.page == case.expected_page)
        and all(word.casefold() in result.text.casefold() for word in case.expected_keywords)
    )


def _answer_checks(
    case: EvaluationCase, answer: GroundedAnswer, results: Sequence[RetrievalResult]
) -> dict[str, bool]:
    unique_results = list(dict.fromkeys(result.chunk_id for result in results))
    by_chunk = {result.chunk_id: result for result in results}
    id_to_chunk = {f"S{position}": chunk_id for position, chunk_id in enumerate(unique_results, 1)}
    inline_ids = set(_INLINE_ID.findall(answer.answer))
    declared_ids = set(answer.evidence_ids)
    citation_ids = {citation.evidence_id for citation in answer.citations}
    valid_citations = (
        len(answer.evidence_ids) == len(declared_ids)
        and len(answer.citations) == len(citation_ids)
        and declared_ids == citation_ids == inline_ids
        and all(evidence_id in id_to_chunk for evidence_id in declared_ids)
    )
    citations_refer_to_retrieved = all(
        citation.evidence_id in id_to_chunk
        and citation.chunk_id == id_to_chunk[citation.evidence_id]
        and citation.chunk_id in by_chunk
        and citation.source == by_chunk[citation.chunk_id].source
        and citation.page == by_chunk[citation.chunk_id].page
        and citation.section == by_chunk[citation.chunk_id].section
        and citation.text == by_chunk[citation.chunk_id].text
        for citation in answer.citations
    )
    valid_citations = valid_citations and citations_refer_to_retrieved
    if case.answerable:
        valid_citations = valid_citations and not answer.insufficient_evidence and bool(declared_ids)
    else:
        valid_citations = valid_citations and answer.insufficient_evidence and not declared_ids
    return {
        "valid_citations": valid_citations,
        "citations_refer_to_retrieved": citations_refer_to_retrieved,
        "insufficient_evidence_correct": answer.insufficient_evidence == (not case.answerable),
        "answer_contains_expected_keywords": (
            all(word.casefold() in answer.answer.casefold() for word in case.expected_keywords)
            if case.answerable
            else answer.insufficient_evidence
        ),
    }


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    index: VectorIndex,
    *,
    top_k: int = 5,
    retrieval_fn: Callable[..., list[RetrievalResult]] = retrieve,
    answer_fn: Callable[..., GroundedAnswer] | None = generate_answer,
) -> EvaluationReport:
    """Run fixed questions; no judge model or semantic grading is used."""
    if not cases:
        raise ValueError("Provide at least one evaluation case.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    rows: list[CaseResult] = []
    for case in cases:
        results = retrieval_fn(case.question, index, top_k=top_k)[:top_k]
        matching_sources = [
            result.rank for result in results if result.source == case.expected_source
        ] if case.answerable else []
        matching_passages = [
            result.rank for result in results if _correct_passage(case, result)
        ] if case.answerable else []
        checks = (
            _answer_checks(case, answer_fn(case.question, results), results)
            if answer_fn is not None
            else {}
        )
        rows.append(
            CaseResult(
                question=case.question,
                answerable=case.answerable,
                expected_source=case.expected_source,
                source_in_top_k=bool(matching_sources) if case.answerable else None,
                source_rank=min(matching_sources) if matching_sources else None,
                correct_result_rank=min(matching_passages) if matching_passages else None,
                hit_at_k=bool(matching_passages) if case.answerable else None,
                **checks,
            )
        )
    answerable_rows = [row for row in rows if row.answerable]
    answer_rows = [row for row in rows if row.valid_citations is not None]
    hits = sum(row.hit_at_k is True for row in answerable_rows)
    return EvaluationReport(
        top_k=top_k,
        cases=rows,
        retrieval_hits=hits,
        retrieval_cases=len(answerable_rows),
        hit_at_k=hits / len(answerable_rows) if answerable_rows else None,
        valid_citation_cases=(
            sum(row.valid_citations is True for row in answer_rows)
            if answer_rows else None
        ),
        abstention_cases_passed=(
            sum(row.insufficient_evidence_correct is True for row in answer_rows if not row.answerable)
            if answer_rows else None
        ),
        answer_cases=len(answer_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local sample RAG corpus.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    suite, documents = load_suite(args.cases)
    settings = load_config()
    index = build_index(
        chunk_documents(parse_documents(documents, config=settings), config=settings),
        config=settings,
    )
    try:
        report = evaluate_cases(
            suite.cases,
            index,
            top_k=args.top_k,
            answer_fn=None if args.retrieval_only else generate_answer,
        )
        print(report.model_dump_json(indent=2))
    finally:
        reset_index(index)


if __name__ == "__main__":
    main()
