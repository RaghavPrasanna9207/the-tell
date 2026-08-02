"""Local LLM client: schema-constrained generation via Ollama.

The whole point of this module is that malformed output is IMPOSSIBLE, not
just unlikely. We pass the target Pydantic model's JSON schema to Ollama's
`format` parameter, which constrains the sampler at the token level. There
is deliberately no JSON-repair or retry-until-parseable loop here — if this
raises, the schema itself is wrong, not the model's output.
"""

import time
from dataclasses import dataclass
from typing import TypeVar

import ollama
from pydantic import BaseModel

TEACHER_MODEL = "qwen2.5:7b-instruct-q4_K_M"

T = TypeVar("T", bound=BaseModel)


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server can't be reached, or the pulled
    model isn't present. Callers should surface this as a clear 503, not
    fall back to fabricated output — see CLAUDE.md: fully local, no fallback
    to a cloud API, ever."""


@dataclass
class GenerationResult:
    text: str
    elapsed_seconds: float
    model: str


def _client() -> ollama.Client:
    return ollama.Client()


def check_available(model: str = TEACHER_MODEL) -> None:
    """Raise OllamaUnavailableError with an actionable message if the
    server isn't running or the model hasn't been pulled."""
    try:
        client = _client()
        tags = client.list()
    except Exception as exc:  # connection refused, etc.
        raise OllamaUnavailableError(
            "Ollama server is not reachable at http://localhost:11434. "
            "Start it with `ollama serve` (see README setup)."
        ) from exc

    available_names = {m.model for m in tags.models}
    if model not in available_names and not any(model.split(":")[0] in n for n in available_names):
        raise OllamaUnavailableError(
            f"Model '{model}' is not pulled. Run `ollama pull {model}` (see README setup)."
        )


def generate_structured(
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    model: str = TEACHER_MODEL,
) -> tuple[T, GenerationResult]:
    """Generate output constrained to `schema`'s JSON schema, then validate
    it into an instance of `schema`. Raises OllamaUnavailableError if the
    server/model isn't available, or pydantic.ValidationError if the model
    (very unusually, given constrained decoding) produces something that
    doesn't fit — that should be rare enough to be worth investigating, not
    silently retried.
    """
    check_available(model)

    client = _client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.monotonic()
    response = client.chat(
        model=model,
        messages=messages,
        format=schema.model_json_schema(),
        options={"temperature": 0.1},
    )
    elapsed = time.monotonic() - start

    content = response["message"]["content"]
    parsed = schema.model_validate_json(content)

    return parsed, GenerationResult(text=content, elapsed_seconds=elapsed, model=model)
