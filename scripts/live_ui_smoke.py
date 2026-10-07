"""Run a small real-service smoke test through Streamlit's UI test harness.

This uses the local .env API key. Run with: python scripts/live_ui_smoke.py
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def check(app: AppTest, stage: str) -> None:
    if app.exception:
        raise RuntimeError(f"{stage}: Streamlit exception: {app.exception[0].value}")
    if app.error:
        raise RuntimeError(f"{stage}: UI error: {app.error[0].value}")


def main() -> None:
    app = AppTest.from_file(ROOT / "app.py", default_timeout=180).run()
    check(app, "initial load")
    assert button(app, "Ask Question").disabled
    print("initial load: OK; Ask disabled")

    content = (ROOT / "evaluation" / "policy.md").read_bytes()
    app.file_uploader[0].upload("policy.md", content, "text/markdown").run()
    check(app, "upload")
    button(app, "Process Documents").click().run()
    check(app, "build")
    assert app.session_state.active_index is not None
    assert app.session_state.active_index.chunk_count > 0
    print(f"build: OK; {app.session_state.active_index.chunk_count} chunks")

    app.text_area(key="question").set_value(
        "How many days of annual leave do employees receive?"
    ).run()
    check(app, "question entry")
    assert not button(app, "Ask Question").disabled
    button(app, "Ask Question").click().run()
    check(app, "answer")
    answer = app.session_state.last_answer
    assert answer is not None and not answer.insufficient_evidence
    assert answer.citations
    assert "20" in answer.answer
    assert all(citation.source == "policy.md" for citation in answer.citations)
    assert app.session_state.last_retrieved
    print("answer: OK; grounded answer and citations returned")
    print("answer text:", answer.answer)
    print("citations:", [citation.evidence_id for citation in answer.citations])
    print("supporting passages:", len(app.session_state.last_retrieved))

    button(app, "Reset").click().run()
    check(app, "reset")
    assert app.session_state.active_index is None
    assert button(app, "Ask Question").disabled
    print("reset: OK; Ask disabled")


if __name__ == "__main__":
    main()
