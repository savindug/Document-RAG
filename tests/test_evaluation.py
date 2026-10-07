"""Deterministic checks for retrieval metrics and citation validation."""

from pathlib import Path

import pytest

from document_rag.chunking import chunk_documents
from document_rag.evaluation import EvaluationCase, _correct_passage, evaluate_cases, load_suite
from document_rag.models import AnswerCitation, GroundedAnswer, RetrievalResult
from document_rag.parsing import parse_documents


def _hit(rank: int, source: str, text: str, page: int | None = None) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        chunk_id=f"{source}:{rank}",
        text=text,
        source=source,
        page=page,
        section="Policy",
        distance=0.1 * rank,
    )


def _answer(result: RetrievalResult) -> GroundedAnswer:
    return GroundedAnswer(
        answer="The answer is 20 days [S2].",
        evidence_ids=["S2"],
        citations=[
            AnswerCitation(
                evidence_id="S2",
                chunk_id=result.chunk_id,
                source=result.source,
                page=result.page,
                section=result.section,
                text=result.text,
            )
        ],
    )


def test_local_fixture_has_answerable_and_unanswerable_questions() -> None:
    suite, documents = load_suite(Path(__file__).resolve().parents[1] / "evaluation" / "cases.json")
    assert len(documents) == 3
    assert len(suite.cases) >= 4
    assert any(not case.answerable for case in suite.cases)
    assert all(case.expected_source in suite.documents for case in suite.cases if case.answerable)
    chunks = chunk_documents(parse_documents(documents))
    assert all(
        any(
            _correct_passage(
                case,
                RetrievalResult(
                    rank=1,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source_name,
                    page=chunk.page_number,
                ),
            )
            for chunk in chunks
        )
        for case in suite.cases
        if case.answerable
    )


def test_retrieval_rank_hit_at_k_and_answer_checks() -> None:
    cases = [
        EvaluationCase(
            question="How much leave?",
            expected_source="policy.pdf",
            expected_page=4,
            expected_keywords=["20 days", "annual leave"],
            answerable=True,
        ),
        EvaluationCase(question="What is the CEO address?", answerable=False),
    ]
    wrong = _hit(1, "policy.pdf", "Remote work is allowed.", page=4)
    correct = _hit(2, "policy.pdf", "Employees receive 20 days of annual leave.", page=4)

    def retrieve_fake(question, index, *, top_k):
        return [wrong, correct] if "leave" in question else [wrong]

    def answer_fake(question, results):
        if "leave" in question:
            return _answer(correct)
        return GroundedAnswer(answer="Insufficient evidence.", insufficient_evidence=True)

    report = evaluate_cases(cases, object(), top_k=2, retrieval_fn=retrieve_fake, answer_fn=answer_fake)
    first, second = report.cases
    assert first.source_in_top_k
    assert first.source_rank == 1
    assert first.correct_result_rank == 2
    assert first.hit_at_k
    assert first.valid_citations
    assert first.citations_refer_to_retrieved
    assert first.insufficient_evidence_correct
    assert second.hit_at_k is None
    assert second.insufficient_evidence_correct
    assert report.hit_at_k == 1
    assert report.valid_citation_cases == 2
    assert report.abstention_cases_passed == 1


def test_wrong_page_or_missing_keyword_is_not_a_retrieval_hit() -> None:
    case = EvaluationCase(
        question="How much leave?",
        expected_source="policy.pdf",
        expected_page=4,
        expected_keywords=["20 days", "annual leave"],
        answerable=True,
    )
    result = _hit(1, "policy.pdf", "Employees receive 20 days of annual leave.", page=5)
    report = evaluate_cases(
        [case], object(), retrieval_fn=lambda *args, **kwargs: [result], answer_fn=None
    )
    assert report.cases[0].source_in_top_k
    assert report.cases[0].correct_result_rank is None
    assert not report.cases[0].hit_at_k
    assert report.hit_at_k == 0
    assert report.valid_citation_cases is None


def test_citation_must_refer_to_supplied_chunk_and_metadata() -> None:
    case = EvaluationCase(
        question="How much leave?",
        expected_source="policy.pdf",
        expected_keywords=["20 days"],
        answerable=True,
    )
    retrieved = _hit(1, "policy.pdf", "Employees receive 20 days of leave.")
    invented = _hit(2, "invented.pdf", "Employees receive 20 days of leave.")
    bad_answer = GroundedAnswer(
        answer="20 days [S1].",
        evidence_ids=["S1"],
        citations=[
            AnswerCitation(
                evidence_id="S1",
                chunk_id=invented.chunk_id,
                source=invented.source,
                text=invented.text,
            )
        ],
    )
    report = evaluate_cases(
        [case],
        object(),
        retrieval_fn=lambda *args, **kwargs: [retrieved],
        answer_fn=lambda *args: bad_answer,
    )
    assert not report.cases[0].valid_citations
    assert not report.cases[0].citations_refer_to_retrieved
    assert report.valid_citation_cases == 0


def test_answerable_case_needs_source_and_keywords() -> None:
    with pytest.raises(ValueError, match="expected source and keywords"):
        EvaluationCase(question="How much leave?", answerable=True)


def test_unanswerable_case_fails_if_answer_is_returned() -> None:
    case = EvaluationCase(question="What is the CEO address?", answerable=False)
    result = _hit(1, "policy.md", "Annual leave is 20 days.")
    answer = GroundedAnswer(
        answer="The answer is 20 days [S1].",
        evidence_ids=["S1"],
        citations=[
            AnswerCitation(
                evidence_id="S1",
                chunk_id=result.chunk_id,
                source=result.source,
                text=result.text,
            )
        ],
    )
    report = evaluate_cases(
        [case],
        object(),
        retrieval_fn=lambda *args, **kwargs: [result],
        answer_fn=lambda *args: answer,
    )
    assert not report.cases[0].insufficient_evidence_correct
    assert not report.cases[0].valid_citations
    assert report.abstention_cases_passed == 0
