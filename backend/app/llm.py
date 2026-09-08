"""Local LLM client: schema-constrained generation via Ollama.

The whole point of this module is that malformed output is IMPOSSIBLE, not
just unlikely. We pass the target Pydantic model's JSON schema to Ollama's
`format` parameter, which constrains the sampler at the token level. There
is deliberately no JSON-repair or retry-until-parseable loop here — if this
raises, the schema itself is wrong, not the model's output.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from app import config

TEACHER_MODEL = config.TEACHER_MODEL

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the configured teacher backend can't be reached or isn't
    usable. Callers should surface this as a clear 503 rather than falling
    back to fabricated output: a scam-explanation tool that invents an
    explanation when its model is down is worse than one that says it's
    down."""


class OllamaUnavailableError(LLMUnavailableError):
    """The Ollama-specific case. Kept as its own name because call sites and
    tests reference it directly; catch LLMUnavailableError to cover both
    backends."""


class NIMUnavailableError(LLMUnavailableError):
    """The NIM-specific case: no API key, auth rejected, or rate limited."""


@dataclass
class GenerationResult:
    text: str
    elapsed_seconds: float
    model: str
    clamped_fields: list[str] = field(default_factory=list)
    """Dotted field paths (e.g. "detections.0.confidence") whose value was
    out of its declared Field(ge=..., le=...) range and got clamped to the
    nearest bound rather than raising. Non-empty only in the rare case
    documented on `_clamp_range_violations`. Empty on every normal call."""


def _client() -> ollama.Client:
    # Host from config, not the library default: under Docker Compose the
    # server is at http://ollama:11434, not localhost.
    return ollama.Client(host=config.OLLAMA_HOST)


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


def _clamp_range_violations(raw: dict, errors: list[dict]) -> list[str] | None:
    """Self-heal the one thing schema-constrained decoding does NOT
    guarantee: numeric fields land within their declared Field(ge=..., le=...)
    bounds. The sampler enforces JSON *shape* (e.g. `confidence` is a float),
    not the *range* Pydantic separately checks — in practice the teacher
    occasionally emits a confidence like 1.3 (seen ~0.2-5% of calls). That's
    a values-out-of-range problem, not a malformed-JSON problem, so clamping
    it here is not the "JSON repair" this module's docstring rules out.

    Returns the list of dotted-path fields that were clamped, or None if any
    error isn't a plain ge/le violation — in that case the caller re-raises,
    since a shape-level validation failure is a real bug worth investigating.
    """
    if not errors or not all(e["type"] in ("greater_than_equal", "less_than_equal") for e in errors):
        return None

    clamped: list[str] = []
    for err in errors:
        loc = err["loc"]
        bound = err["ctx"]["ge"] if err["type"] == "greater_than_equal" else err["ctx"]["le"]
        target = raw
        for key in loc[:-1]:
            target = target[key]
        target[loc[-1]] = bound
        clamped.append(".".join(str(p) for p in loc))
    return clamped


def generate_structured(
    prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
    model: str = TEACHER_MODEL,
) -> tuple[T, GenerationResult]:
    """Generate output constrained to `schema`'s JSON schema, then validate
    it into an instance of `schema`. Raises OllamaUnavailableError if the
    server/model isn't available. A ge/le range violation (see
    `_clamp_range_violations`) is clamped and logged rather than raised;
    any other pydantic.ValidationError means the model (very unusually,
    given constrained decoding) produced something structurally unexpected
    — that's rare enough to be worth investigating, not silently retried.
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
    clamped_fields: list[str] = []
    try:
        parsed = schema.model_validate_json(content)
    except ValidationError as exc:
        raw = json.loads(content)
        clamped_fields = _clamp_range_violations(raw, exc.errors()) or []
        if not clamped_fields:
            raise
        logger.warning("Clamped out-of-range field(s) %s in %s output", clamped_fields, model)
        parsed = schema.model_validate(raw)

    return parsed, GenerationResult(
        text=content, elapsed_seconds=elapsed, model=model, clamped_fields=clamped_fields
    )
