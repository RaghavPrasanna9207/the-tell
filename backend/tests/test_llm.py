"""Tests for the schema-constrained generation boundary in app/llm.py,
specifically the out-of-range confidence self-heal (see
`_clamp_range_violations`'s docstring): schema-constrained decoding
guarantees JSON *shape*, not declared numeric *range*, and the teacher
occasionally emits a confidence outside [0, 1]. That must not become an
unhandled 500 on /analyze.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.llm import generate_structured
from app.taxonomy import Analysis


def _fake_chat_response(content: str) -> dict:
    return {"message": {"content": content}}


def _mock_client(content: str) -> MagicMock:
    client = MagicMock()
    client.list.return_value = MagicMock(models=[MagicMock(model="qwen2.5:7b-instruct-q4_K_M")])
    client.chat.return_value = _fake_chat_response(content)
    return client


def test_out_of_range_confidence_is_clamped_not_raised():
    """A confidence of 1.3 (structurally valid JSON, out-of-spec value) must
    be clamped to 1.0 and returned, not raised as a ValidationError."""
    content = (
        '{"detections": [{"technique": "isolation", "span": "x", "confidence": 1.3}]}'
    )
    with patch("app.llm._client", return_value=_mock_client(content)):
        parsed, generation = generate_structured(prompt="msg", schema=Analysis)

    assert parsed.detections[0].confidence == 1.0
    assert generation.clamped_fields == ["detections.0.confidence"]


def test_negative_confidence_is_clamped_to_zero():
    content = (
        '{"detections": [{"technique": "isolation", "span": "x", "confidence": -0.2}]}'
    )
    with patch("app.llm._client", return_value=_mock_client(content)):
        parsed, generation = generate_structured(prompt="msg", schema=Analysis)

    assert parsed.detections[0].confidence == 0.0
    assert generation.clamped_fields == ["detections.0.confidence"]


def test_in_range_confidence_is_not_touched():
    content = '{"detections": [{"technique": "isolation", "span": "x", "confidence": 0.9}]}'
    with patch("app.llm._client", return_value=_mock_client(content)):
        parsed, generation = generate_structured(prompt="msg", schema=Analysis)

    assert parsed.detections[0].confidence == 0.9
    assert generation.clamped_fields == []


def test_non_range_validation_error_still_raises():
    """A structurally wrong field (missing `span` entirely) is a real schema
    problem, not a range violation — must still raise, not be silently
    swallowed."""
    content = '{"detections": [{"technique": "isolation", "confidence": 0.9}]}'
    with patch("app.llm._client", return_value=_mock_client(content)):
        with pytest.raises(ValidationError):
            generate_structured(prompt="msg", schema=Analysis)
