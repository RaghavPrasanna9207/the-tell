"""End-to-end test of the /analyze route, mocked only at the LLM boundary
(app.llm.generate_structured). This exercises the real wiring between
main.py, classify.py, and explain.py — unlike the per-module unit tests,
it would catch a mismatched field name or a schema drift between them.

Every test that needs classify() to actually run patches the Layer-1
cascade gate (app.main.student_available / app.main.should_investigate) to
force it open. Without that, these tests would be at the mercy of whatever
the real student model (present on this dev machine at
backend/models/student) happens to decide about the exact test string,
which is exactly the nondeterminism a test suite can't have. The gate's
own behavior is covered separately below and in test_student.py.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.llm import GenerationResult, OllamaUnavailableError
from app.main import app
from app.taxonomy import Analysis, Detection, Technique

client = TestClient(app)


def _fake_generation() -> GenerationResult:
    return GenerationResult(text="{}", elapsed_seconds=0.01, model="test-model")


def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_analyze_returns_503_when_ollama_unavailable():
    """Forces the OllamaUnavailableError path (rather than relying on no
    Ollama server being reachable in the test environment, which doesn't
    hold when tests run on a dev machine with Ollama running) and asserts
    the app surfaces a clean 503, never falling back to fabricated output."""
    with (
        patch("app.main.student_available", return_value=True),
        patch("app.main.should_investigate", return_value=True),
        patch(
            "app.llm.check_available",
            side_effect=OllamaUnavailableError("Ollama server is not reachable at http://localhost:11434."),
        ),
    ):
        r = client.post("/analyze", json={"text": "some message"})
    assert r.status_code == 503
    assert "ollama" in r.json()["detail"].lower() or "not reachable" in r.json()["detail"].lower()


def test_analyze_clean_message_end_to_end():
    message = "Your OTP is 483920. Valid for 10 minutes."
    with (
        patch("app.main.student_available", return_value=True),
        patch("app.main.should_investigate", return_value=True),
        patch("app.classify.generate_structured", return_value=(Analysis(), _fake_generation())),
    ):
        r = client.post("/analyze", json={"text": message})

    assert r.status_code == 200
    body = r.json()
    assert body["is_clean"] is True
    assert body["cards"] == []


def test_analyze_scam_message_end_to_end():
    """Mocks both LLM call sites (classify's detection + explain's purpose
    clause) to verify the full pipeline assembles a correct response: the
    card's explanation must contain the verbatim span (inserted
    programmatically, not by the mocked "LLM" purpose clause) and the
    exact, unmodified reality-check/source from the counter-move table."""
    message = "Do not tell your family. This is CBI calling."
    span = "Do not tell your family"

    fake_analysis = Analysis(
        detections=[Detection(technique=Technique.ISOLATION, span=span, confidence=0.92)]
    )

    fake_purpose = "This exists to stop anyone from stepping in before you act."

    with (
        patch("app.main.student_available", return_value=True),
        patch("app.main.should_investigate", return_value=True),
        patch("app.classify.generate_structured", return_value=(fake_analysis, _fake_generation())),
        patch(
            "app.explain._generate_purpose_clause",
            return_value=(fake_purpose, _fake_generation()),
        ),
    ):
        r = client.post("/analyze", json={"text": message})

    assert r.status_code == 200
    body = r.json()
    assert body["is_clean"] is False
    assert len(body["cards"]) == 1

    card = body["cards"][0]
    assert card["technique"] == "isolation"
    assert card["span"] == span
    assert span in card["explanation"]

    from app.counter_moves import get_counter_move

    counter_move = get_counter_move(Technique.ISOLATION)
    assert counter_move.reality_check.strip() in card["explanation"]
    assert card["source"] == counter_move.source


def test_analyze_survives_out_of_range_teacher_confidence():
    """Regression test for a real observed failure: the teacher occasionally
    emits a confidence outside [0, 1] (schema-constrained decoding enforces
    JSON shape, not declared numeric range). That must be clamped inside
    app.llm.generate_structured and never surface as a 500 on /analyze. This
    mocks at the raw Ollama-client boundary (app.llm._client), not
    app.classify.generate_structured, so the real clamp logic executes."""
    detect_content = (
        '{"detections": [{"technique": "isolation", '
        '"span": "Do not tell your family", "confidence": 1.3}]}'
    )
    explain_content = '{"purpose": "This exists to stop anyone from stepping in before you act."}'

    fake_client = MagicMock()
    fake_client.list.return_value = MagicMock(models=[MagicMock(model="qwen2.5:7b-instruct-q4_K_M")])
    fake_client.chat.side_effect = [
        {"message": {"content": detect_content}},
        {"message": {"content": explain_content}},
    ]

    with (
        patch("app.main.student_available", return_value=True),
        patch("app.main.should_investigate", return_value=True),
        patch("app.llm._client", return_value=fake_client),
    ):
        r = client.post("/analyze", json={"text": "Do not tell your family. This is CBI calling."})

    assert r.status_code == 200
    card = r.json()["cards"][0]
    assert card["confidence"] == 1.0


def test_gate_skips_teacher_when_student_finds_nothing():
    """The whole point of the Phase A1 cascade: when the student gate says
    there's nothing to investigate, /analyze must return clean WITHOUT ever
    calling the teacher. Rigs classify's LLM call site to blow up if
    reached, so this fails loudly (not just "coincidentally passes") if the
    short-circuit ever breaks."""
    with (
        patch("app.main.student_available", return_value=True),
        patch("app.main.should_investigate", return_value=False),
        patch(
            "app.classify.generate_structured",
            side_effect=AssertionError("teacher should not have been called"),
        ),
    ):
        r = client.post("/analyze", json={"text": "Your OTP is 483920."})

    assert r.status_code == 200
    body = r.json()
    assert body["is_clean"] is True
    assert body["cards"] == []


def test_gate_wakes_teacher_when_student_flags_something():
    """When the gate says investigate, the existing classify/explain flow
    must run exactly as it did before the cascade existed."""
    span = "Do not tell your family"
    fake_analysis = Analysis(
        detections=[Detection(technique=Technique.ISOLATION, span=span, confidence=0.92)]
    )
    fake_purpose = "This exists to stop anyone from stepping in before you act."

    with (
        patch("app.main.student_available", return_value=True),
        patch("app.main.should_investigate", return_value=True),
        patch("app.classify.generate_structured", return_value=(fake_analysis, _fake_generation())),
        patch("app.explain._generate_purpose_clause", return_value=(fake_purpose, _fake_generation())),
    ):
        r = client.post("/analyze", json={"text": "Do not tell your family. This is CBI calling."})

    assert r.status_code == 200
    body = r.json()
    assert body["is_clean"] is False
    assert body["cards"][0]["technique"] == "isolation"


def test_gate_is_skipped_when_student_model_absent():
    """A fresh clone that has never run eval/distill.py has no student model
    on disk — the gate must degrade to "always call the teacher", not
    crash. See app.student.is_available's docstring."""
    with (
        patch("app.main.student_available", return_value=False),
        patch("app.classify.generate_structured", return_value=(Analysis(), _fake_generation())),
    ):
        r = client.post("/analyze", json={"text": "Your OTP is 483920."})

    assert r.status_code == 200
    assert r.json()["is_clean"] is True


def test_analyze_rejects_empty_text():
    r = client.post("/analyze", json={"text": ""})
    assert r.status_code == 422  # pydantic min_length validation


def test_analyze_rejects_oversized_text():
    r = client.post("/analyze", json={"text": "x" * 5000})
    assert r.status_code == 422  # pydantic max_length validation
