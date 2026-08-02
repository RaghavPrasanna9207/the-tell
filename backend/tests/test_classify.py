"""Tests for Layer 1's span-verification logic.

Constrained decoding guarantees the LLM's output is valid JSON shape — it
does NOT guarantee that a `span` field is an actual verbatim substring of
the input. `classify()` must catch and drop any detection whose span isn't
literally present, since that's exactly the kind of grounding failure
Layer 2 depends on not happening. Mocked at the `generate_structured` call
so no live Ollama connection is needed.
"""

from unittest.mock import patch

from app.classify import classify
from app.llm import GenerationResult
from app.taxonomy import Analysis, Detection, Technique


def _fake_generation() -> GenerationResult:
    return GenerationResult(text="{}", elapsed_seconds=0.01, model="test-model")


def test_all_valid_spans_are_kept():
    message = "Do not tell your family. Transfer within 2 hours."
    fake_analysis = Analysis(
        detections=[
            Detection(technique=Technique.ISOLATION, span="Do not tell your family", confidence=0.9),
            Detection(technique=Technique.MANUFACTURED_URGENCY, span="within 2 hours", confidence=0.85),
        ]
    )
    with patch("app.classify.generate_structured", return_value=(fake_analysis, _fake_generation())):
        result = classify(message)

    assert len(result.analysis.detections) == 2
    assert result.dropped_spans == []


def test_hallucinated_span_is_dropped_not_trusted():
    """The model claims a span that isn't actually in the message — this
    must be dropped, not silently kept, or Layer 2's "quote the exact
    words" guarantee breaks downstream."""
    message = "Your parcel contains drugs. Call this number now."
    fake_analysis = Analysis(
        detections=[
            Detection(technique=Technique.FALSE_AUTHORITY, span="CBI officer speaking", confidence=0.8),
            Detection(technique=Technique.CHANNEL_SWITCH, span="Call this number", confidence=0.9),
        ]
    )
    with patch("app.classify.generate_structured", return_value=(fake_analysis, _fake_generation())):
        result = classify(message)

    kept_techniques = {d.technique for d in result.analysis.detections}
    assert Technique.CHANNEL_SWITCH in kept_techniques
    assert Technique.FALSE_AUTHORITY not in kept_techniques
    assert result.dropped_spans == ["CBI officer speaking"]


def test_clean_message_produces_no_detections():
    message = "Your OTP is 483920. Valid for 10 minutes."
    with patch("app.classify.generate_structured", return_value=(Analysis(), _fake_generation())):
        result = classify(message)

    assert result.analysis.is_clean is True
    assert result.dropped_spans == []
