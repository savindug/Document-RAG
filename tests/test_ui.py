"""UI lifecycle tests without Gemini or a real Chroma collection."""

from types import SimpleNamespace
from pathlib import Path

from streamlit.testing.v1 import AppTest
from pydantic import ValidationError

from document_rag import ui
from document_rag.models import AnswerCitation, DocumentChunk, GroundedAnswer, ParsedDocument, ParsedElement, RetrievalResult, UploadedDocument


def _parsed(name: str) -> ParsedDocument:
    return ParsedDocument(
        source_id=name,
        source_name=name,
        elements=[
            ParsedElement(
                element_id=f"{name}:0",
                source_id=name,
                source_name=name,
                text="Annual leave is 20 days.",
                category="NarrativeText",
                page_number=2,
                section_title="Leave",
            )
        ],
    )


def _chunk(name: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{name}:chunk",
        source_id=name,
        source_name=name,
        text="Leave: Annual leave is 20 days.",
        embedding_text="Leave: Annual leave is 20 days.",
        page_number=2,
        heading="Leave",
    )


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def test_ui_build_ask_replace_and_reset(monkeypatch) -> None:
    indexes = []
    deleted = []

    monkeypatch.setattr(
        ui,
        "load_config",
        lambda: SimpleNamespace(
            require_google_api_key=lambda: "fake",
            unstructured_api_key=None,
            unstructured_api_url=None,
        ),
    )
    monkeypatch.setattr(ui, "parse_documents", lambda uploads, config: [_parsed(item.filename) for item in uploads])
    monkeypatch.setattr(ui, "chunk_documents", lambda parsed, config: [_chunk(item.source_name) for item in parsed])

    def fake_build(chunks, *, config, previous_index):
        if chunks[0].source_name == "fail.txt":
            raise RuntimeError("candidate failed")
        index = SimpleNamespace(chunk_count=len(chunks), ready=True, name=f"index-{len(indexes)}")
        indexes.append(index)
        if previous_index is not None:
            previous_index.ready = False
            deleted.append(previous_index.name)
        return index

    monkeypatch.setattr(ui, "build_index", fake_build)
    monkeypatch.setattr(ui, "is_index_ready", lambda index: bool(index and index.ready))
    monkeypatch.setattr(ui, "reset_index", lambda index: (deleted.append(index.name), setattr(index, "ready", False)))
    monkeypatch.setattr(
        ui,
        "retrieve",
        lambda question, index, config: [
            RetrievalResult(
                rank=1,
                chunk_id="policy.txt:chunk",
                text="Leave: Annual leave is 20 days.",
                source="policy.txt",
                page=2,
                section="Leave",
                distance=0.1,
            )
        ],
    )
    monkeypatch.setattr(
        ui,
        "generate_answer",
        lambda question, results, config: GroundedAnswer(
            answer="Annual leave is 20 days [S1].",
            evidence_ids=["S1"],
            citations=[
                AnswerCitation(
                    evidence_id="S1",
                    chunk_id=results[0].chunk_id,
                    source=results[0].source,
                    page=results[0].page,
                    section=results[0].section,
                    text=results[0].text,
                )
            ],
        ),
    )

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert _button(app, "Ask Question").disabled
    app.file_uploader[0].upload("policy.txt", b"Annual leave is 20 days.", "text/plain").run()
    _button(app, "Process Documents").click().run()
    assert not app.exception
    assert app.session_state.active_index is indexes[0]
    assert app.session_state.active_filenames == ["policy.txt"]
    assert not _button(app, "Ask Question").disabled
    _button(app, "Ask Question").click().run()
    assert any("Enter a question" in warning.value for warning in app.warning)
    assert app.session_state.last_answer is None

    app.text_area(key="question").set_value("How much leave?").run()
    assert not _button(app, "Ask Question").disabled
    _button(app, "Ask Question").click().run()
    assert app.session_state.last_answer.answer == "Annual leave is 20 days [S1]."
    assert app.session_state.last_retrieved[0].distance == 0.1
    assert any("policy.txt" in item.label for item in app.expander)
    assert any(item.label == "Debug" for item in app.expander)
    assert len(app.dataframe) == 3

    app.file_uploader[0].set_value(("policy.txt", b"Changed upload content", "text/plain")).run()
    assert not _button(app, "Ask Question").disabled
    assert any("Selected files differ" in warning.value for warning in app.warning)
    assert app.session_state.active_filenames == ["policy.txt"]

    app.file_uploader[0].set_value(("fail.txt", b"candidate", "text/plain")).run()
    _button(app, "Process Documents").click().run()
    assert app.session_state.active_index is indexes[0]
    assert indexes[0].ready
    assert not _button(app, "Ask Question").disabled
    assert app.session_state.active_filenames == ["policy.txt"]
    _button(app, "Ask Question").click().run()
    assert app.session_state.last_answer.answer == "Annual leave is 20 days [S1]."

    app.file_uploader[0].set_value(("new.md", b"# New", "text/markdown")).run()
    assert not _button(app, "Ask Question").disabled
    _button(app, "Process Documents").click().run()
    assert app.session_state.active_index is indexes[1]
    assert deleted == ["index-0"]
    assert app.session_state.last_answer is None
    assert app.session_state.active_filenames == ["new.md"]

    _button(app, "Reset").click().run()
    assert _button(app, "Ask Question").disabled
    assert app.session_state.active_index is None
    assert app.session_state.chunks == []
    assert deleted == ["index-0", "index-1"]


def test_upload_validation_error_does_not_echo_file_content() -> None:
    try:
        UploadedDocument(filename="bad.png", content=b"private-upload-data")
    except ValidationError as exc:
        message = ui._validation_message(exc)
    else:
        raise AssertionError("Expected validation error")
    assert "supported" in message
    assert "private-upload-data" not in message


def test_ui_processes_multiple_selected_files(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        ui,
        "load_config",
        lambda: SimpleNamespace(unstructured_api_key=None, unstructured_api_url=None),
    )
    monkeypatch.setattr(ui, "is_index_ready", lambda index: bool(index and index.ready))

    def fake_build(files, signature):
        seen.append([file.name for file in files])
        ui.st.session_state.active_index = SimpleNamespace(chunk_count=2, ready=True)
        ui.st.session_state.active_signature = signature
        ui.st.session_state.active_filenames = seen[-1]
        ui.st.session_state.parsed_documents = [_parsed(name) for name in seen[-1]]

    monkeypatch.setattr(ui, "_build", fake_build)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    app.file_uploader[0].set_value([
        ("policy.txt", b"Annual leave is 20 days.", "text/plain"),
        ("benefits.md", b"# Benefits", "text/markdown"),
    ]).run()
    assert not _button(app, "Process Documents").disabled
    _button(app, "Process Documents").click().run()
    assert not app.exception
    assert seen == [["policy.txt", "benefits.md"]]
    assert app.session_state.active_index.chunk_count == 2
    assert not _button(app, "Ask Question").disabled


def test_gemini_service_errors_are_reported_without_response_details() -> None:
    for code, expected in ((503, "temporarily busy"), (429, "quota or rate limit")):
        api_error = RuntimeError("private response details")
        api_error.code = code
        wrapped = RuntimeError("generation failed")
        wrapped.__cause__ = api_error
        message = ui._service_error_message(wrapped, "fallback")
        assert expected in message
        assert "private response details" not in message
