"""Layer 2: grounded explanation generator.

The LLM's only job is to write ONE short clause explaining WHY that
specific manipulation is happening — it never sees, generates, or
paraphrases the reality-check or counter-action, and it never handles the
verbatim quote either. The quote is inserted programmatically from the
Layer-1 span, not asked of the model: constrained decoding guarantees valid
JSON, but not that a small quantized model reliably copies 20+ words of
source text into a field verbatim (in practice it usually paraphrased
instead — see git history for the validation-failure data behind this).
Narrowing the model's job to "phrase the purpose, nothing else" makes the
verbatim-quote guarantee hold by construction, the same way the
reality-check/counter-action text does, not by hoping a validator catches
a drift.

See CLAUDE.md: the model selects and phrases, it never invents.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.counter_moves import CounterMove, get_counter_move
from app.llm import GenerationResult, generate_structured
from app.taxonomy import Detection

MAX_PURPOSE_WORDS = 30

FORBIDDEN_HEDGES = (
    "might be a scam",
    "may be a scam",
    "could be a scam",
    "possibly a scam",
    "perhaps this is a scam",
    "this might",
    "this may",
    "we think this could",
    "seems like it could",
)

SYSTEM_PROMPT = """You explain manipulation techniques in scam messages to \
someone who may currently be under pressure from one.

You will be given the exact words from a message and the name of the \
manipulation technique they demonstrate. Those exact words are already \
shown to the reader elsewhere — your only job is to write ONE short clause \
explaining WHY the scammer is using this specific technique on the reader \
right now: not that the technique is present, but its purpose, what it is \
trying to make the reader do or feel.

Rules:
- Do NOT quote, repeat, or paraphrase the message's exact words back — \
explain their purpose instead, don't restate them.
- Warm, direct, second-person voice. Never condescending. Never "you fell \
for" or "you should have known."
- States facts plainly. No hedging like "might be" or "could possibly be."
- Does NOT include legal facts, statistics, or advice — only the purpose \
of this one technique.
- Is at most {max_words} words.

Example: for manufactured_urgency, a good purpose clause is "The deadline \
exists to stop you from calling your son to check if this is real."
"""


class PurposeClause(BaseModel):
    purpose: str = Field(
        description=(
            "A short clause, warm and direct, explaining the purpose behind "
            "this specific manipulation — why the scammer is using it on the "
            "reader right now. Does not quote or repeat the message text."
        )
    )


@dataclass
class ExplainResult:
    detection: Detection
    counter_move: CounterMove
    explanation: str
    used_fallback: bool
    validation_attempts: int
    generation: GenerationResult | None
    """None only when every LLM attempt failed and we fell back to
    table-only text with no model call succeeding at all."""


def validate_purpose_clause(purpose: str, span: str) -> list[str]:
    """Return validation problems; an empty list means it passed."""
    problems = []
    word_count = len(purpose.split())
    if word_count > MAX_PURPOSE_WORDS:
        problems.append(f"too long ({word_count} words, max {MAX_PURPOSE_WORDS})")
    if span and span.lower() in purpose.lower():
        problems.append("purpose clause repeats the verbatim span instead of explaining its purpose")
    lowered = purpose.lower()
    for phrase in FORBIDDEN_HEDGES:
        if phrase in lowered:
            problems.append(f"contains hedging phrase: {phrase!r}")
    return problems


def _fallback_explanation(detection: Detection, counter_move: CounterMove) -> str:
    """Table-only text, no LLM involved at all — used when the model can't
    produce a valid purpose clause after retrying once."""
    purpose = f'"{detection.span}" is a sign of {counter_move.plain_name.lower()}.'
    return f"{purpose} {counter_move.reality_check.strip()} {counter_move.counter_action.strip()}"


def _generate_purpose_clause(detection: Detection, counter_move: CounterMove) -> tuple[str, GenerationResult]:
    prompt = (
        f"Manipulation technique: {counter_move.plain_name}\n"
        f"Exact words from the message (already shown to the reader, do not repeat them): "
        f'"{detection.span}"\n'
        f"Why this technique generally works: {counter_move.why_it_works}"
    )
    result, generation = generate_structured(
        prompt=prompt,
        schema=PurposeClause,
        system=SYSTEM_PROMPT.format(max_words=MAX_PURPOSE_WORDS),
    )
    return result.purpose, generation


def explain(detection: Detection) -> ExplainResult:
    """Run Layer 2 on a single detection. Raises OllamaUnavailableError if
    the local model isn't running (see app.llm)."""
    counter_move = get_counter_move(detection.technique)

    last_generation: GenerationResult | None = None
    for attempt in range(1, 3):  # one try, one retry
        purpose, generation = _generate_purpose_clause(detection, counter_move)
        last_generation = generation
        problems = validate_purpose_clause(purpose, detection.span)
        if not problems:
            explanation = (
                f'"{detection.span}" — {purpose.strip()} '
                f"{counter_move.reality_check.strip()} {counter_move.counter_action.strip()}"
            )
            return ExplainResult(
                detection=detection,
                counter_move=counter_move,
                explanation=explanation,
                used_fallback=False,
                validation_attempts=attempt,
                generation=generation,
            )

    return ExplainResult(
        detection=detection,
        counter_move=counter_move,
        explanation=_fallback_explanation(detection, counter_move),
        used_fallback=True,
        validation_attempts=2,
        generation=last_generation,
    )


def explain_all(detections: list[Detection]) -> list[ExplainResult]:
    return [explain(d) for d in detections]
