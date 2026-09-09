"""Tests for the NIM backend in app/llm.py.

Mocked at the transport boundary (`httpx.post`), matching how test_llm.py
mocks Ollama at its client factory — no test in this repo touches the network.

What matters here is that switching LLM_BACKEND changes nothing a caller can
observe except which service answers: same signature, same parsing, same
clamping, and failures surface as LLMUnavailableError so app/main.py turns
them into a 503 rather than a 500.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.llm import LLMUnavailableError, NIMUnavailableError, generate_structured
from app.taxonomy import Analysis, Technique

VALID = '{"detections": [{"technique": "isolation", "span": "don\'t tell your family", "confidence": 0.9}]}'


def _mock_response(status_code: int = 200, content: str = VALID, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


def _nim_env(**overrides):
    """Patch app.config as the NIM backend, since llm.py reads it at call time."""
    defaults = {
        "LLM_BACKEND": "nim",
        "NIM_API_KEY": "nvapi-test",
        "NIM_BASE_URL": "https://integrate.api.nvidia.com/v1",
        "NIM_MODEL": "meta/llama-3.3-70b-instruct",
        "NIM_STRUCTURED_MODE": "guided_json",
    }
    defaults.update(overrides)
    return patch.multiple("app.config", **defaults)


def test_nim_sends_the_schema_as_guided_json():
    """The whole reason NIM was chosen over the alternatives: it accepts this
    project's Pydantic schema as-is, so Layer 1's constrained-decoding
    guarantee survives the move off Ollama."""
    with _nim_env(), patch("httpx.post", return_value=_mock_response()) as post:
        parsed, generation = generate_structured(prompt="msg", schema=Analysis)

    body = post.call_args.kwargs["json"]
    assert body["nvext"]["guided_json"] == Analysis.model_json_schema()
    assert body["model"] == "meta/llama-3.3-70b-instruct"
    assert "response_format" not in body
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer nvapi-test"
    assert parsed.detections[0].technique is Technique.ISOLATION
    assert generation.model == "meta/llama-3.3-70b-instruct"


def test_json_object_mode_sends_response_format_instead():
    with _nim_env(NIM_STRUCTURED_MODE="json_object"), patch("httpx.post", return_value=_mock_response()) as post:
        generate_structured(prompt="msg", schema=Analysis)

    body = post.call_args.kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert "nvext" not in body


def test_system_prompt_is_sent_as_a_system_message():
    with _nim_env(), patch("httpx.post", return_value=_mock_response()) as post:
        generate_structured(prompt="msg", schema=Analysis, system="you are a detector")

    messages = post.call_args.kwargs["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "you are a detector"}
    assert messages[1] == {"role": "user", "content": "msg"}


def test_out_of_range_confidence_is_clamped_on_the_nim_path_too():
    """The clamp exists because constrained decoding enforces JSON shape, not
    declared numeric range. That's a property of constrained decoding in
    general, not of Ollama, so it must hold here as well."""
    out_of_range = '{"detections": [{"technique": "isolation", "span": "x", "confidence": 1.3}]}'
    with _nim_env(), patch("httpx.post", return_value=_mock_response(content=out_of_range)):
        parsed, generation = generate_structured(prompt="msg", schema=Analysis)

    assert parsed.detections[0].confidence == 1.0
    assert generation.clamped_fields == ["detections.0.confidence"]


def test_missing_api_key_is_actionable_and_not_a_crash():
    with _nim_env(NIM_API_KEY=""), pytest.raises(NIMUnavailableError, match="build.nvidia.com"):
        generate_structured(prompt="msg", schema=Analysis)


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "API key"), (403, "API key"), (429, "rate limit"), (500, "HTTP 500")],
)
def test_http_failures_surface_as_llm_unavailable(status: int, expected: str):
    """app/main.py catches LLMUnavailableError and returns 503. Anything that
    escapes as a different exception type becomes an unhandled 500 instead."""
    with _nim_env(), patch("httpx.post", return_value=_mock_response(status_code=status, text="upstream boom")):
        with pytest.raises(LLMUnavailableError, match=expected):
            generate_structured(prompt="msg", schema=Analysis)


def test_transport_error_surfaces_as_llm_unavailable():
    import httpx

    with _nim_env(), patch("httpx.post", side_effect=httpx.ConnectError("no route")):
        with pytest.raises(LLMUnavailableError, match="Could not reach NIM"):
            generate_structured(prompt="msg", schema=Analysis)
