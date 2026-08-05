"""End-to-end test of the /analyze route, mocked only at the LLM boundary
(app.llm.generate_structured). This exercises the real wiring between
main.py, classify.py, and explain.py — unlike the per-module unit tests,
it would catch a mismatched field name or a schema drift between them.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.llm import GenerationResult
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
    """No mocking here — this hits the real check_available() path, which
    should fail cleanly since there's no Ollama server in the test
    environment, and must never fall back to fabricated output."""
    r = client.post("/analyze", json={"text": "some message"})
    assert r.status_code == 503
    assert "ollama" in r.json()["detail"].lower() or "not reachable" in r.json()["detail"].lower()


def test_analyze_clean_message_end_to_end():
    message = "Your OTP is 483920. Valid for 10 minutes."
    with patch("app.classify.generate_structured", return_value=(Analysis(), _fake_generation())):
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


def test_analyze_rejects_empty_text():
    r = client.post("/analyze", json={"text": ""})
    assert r.status_code == 422  # pydantic min_length validation


def test_analyze_rejects_oversized_text():
    r = client.post("/analyze", json={"text": "x" * 5000})
    assert r.status_code == 422  # pydantic max_length validation
