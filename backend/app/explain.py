"""Layer 2: grounded explanation generator.

The LLM's only job is to write ONE short sentence that quotes the verbatim
span and explains WHY that specific manipulation is happening — it never
sees, generates, or paraphrases the reality-check or counter-action. Those
are copied byte-for-byte from `data/counter_moves.yaml` and assembled
afterward. This narrows the model's entire risk surface to one sentence,
and means the "no hallucinated facts" guarantee holds by construction, not
by hoping a validator catches a drift.

See CLAUDE.md: the model selects and phrases, it never invents.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.counter_moves import CounterMove, get_counter_move
from app.llm import GenerationResult, generate_structured
from app.taxonomy import Detection

MAX_PURPOSE_WORDS = 40

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
manipulation technique they demonstrate. Write ONE sentence that:
- Quotes the exact words verbatim (copy them exactly, in quotation marks)
- Explains WHY the scammer is using this specific technique on the reader \
right now — not just that the technique is present, but its purpose. \
Example style: "The deadline exists to stop you from calling your son."
- Uses a warm, direct, second-person voice. Never condescending. Never \
"you fell for" or "you should have known."
- States facts plainly. No hedging like "might be" or "could possibly be."
- Does NOT include legal facts, statistics, or advice — only the purpose \
of this one technique.
- Is at most {max_words} words.
"""


class PurposeSentence(BaseModel):
    sentence: str = Field(
        description=(
            "One sentence, warm and direct, quoting the exact span verbatim "
            "and explaining the purpose behind this specific manipulation."
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


def validate_purpose_sentence(sentence: str, span: str) -> list[str]:
    """Return validation problems; an empty list means it passed."""
    problems = []
    if span not in sentence:
        problems.append("missing verbatim span")
    word_count = len(sentence.split())
    if word_count > MAX_PURPOSE_WORDS:
        problems.append(f"too long ({word_count} words, max {MAX_PURPOSE_WORDS})")
    lowered = sentence.lower()
    for phrase in FORBIDDEN_HEDGES:
        if phrase in lowered:
            problems.append(f"contains hedging phrase: {phrase!r}")
    return problems


def _fallback_explanation(detection: Detection, counter_move: CounterMove) -> str:
    """Table-only text, no LLM involved at all — used when the model can't
    produce a valid purpose sentence after retrying once."""
    purpose = f'"{detection.span}" is a sign of {counter_move.plain_name.lower()}.'
    return f"{purpose} {counter_move.reality_check.strip()} {counter_move.counter_action.strip()}"


def _generate_purpose_sentence(detection: Detection, counter_move: CounterMove) -> tuple[str, GenerationResult]:
    prompt = (
        f"Manipulation technique: {counter_move.plain_name}\n"
        f"Exact words from the message: \"{detection.span}\"\n"
        f"Why this technique generally works: {counter_move.why_it_works}"
    )
    result, generation = generate_structured(
        prompt=prompt,
        schema=PurposeSentence,
        system=SYSTEM_PROMPT.format(max_words=MAX_PURPOSE_WORDS),
    )
    return result.sentence, generation


def explain(detection: Detection) -> ExplainResult:
    """Run Layer 2 on a single detection. Raises OllamaUnavailableError if
    the local model isn't running (see app.llm)."""
    counter_move = get_counter_move(detection.technique)

    last_generation: GenerationResult | None = None
    for attempt in range(1, 3):  # one try, one retry
        sentence, generation = _generate_purpose_sentence(detection, counter_move)
        last_generation = generation
        problems = validate_purpose_sentence(sentence, detection.span)
        if not problems:
            explanation = f"{sentence} {counter_move.reality_check.strip()} {counter_move.counter_action.strip()}"
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
