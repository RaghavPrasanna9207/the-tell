"""Layer 1: multi-label manipulation-technique classifier.

Calls the local teacher model (Qwen2.5-7B via Ollama) with schema-constrained
decoding, then enforces the one guarantee constrained decoding does NOT give
you for free: that every `span` is an actual verbatim substring of the
input message, not a paraphrase or a hallucinated quote. Detections that
fail this check are dropped and counted — that count is itself an eval
signal, not just an internal safeguard.
"""

from dataclasses import dataclass, field

from app.llm import GenerationResult, generate_structured
from app.taxonomy import TECHNIQUE_DESCRIPTIONS, Analysis, Technique

SYSTEM_PROMPT = """You are a manipulation-technique detector for a scam-protection tool.

Given a message (SMS, WhatsApp, or similar), identify every manipulation \
technique it uses from the fixed list below. For each one you find, quote \
the EXACT substring from the message that demonstrates it — copy the words \
verbatim, do not paraphrase or summarize. If the message contains no \
manipulation technique, return an empty list. Do not invent techniques that \
aren't in the list, and do not detect a technique unless you can point to \
specific words in the message that demonstrate it.

Techniques:
{technique_list}
"""


def _build_system_prompt() -> str:
    lines = [f"- {t.value}: {TECHNIQUE_DESCRIPTIONS[t]}" for t in Technique]
    return SYSTEM_PROMPT.format(technique_list="\n".join(lines))


@dataclass
class ClassifyResult:
    analysis: Analysis
    generation: GenerationResult
    dropped_spans: list[str] = field(default_factory=list)
    """Spans the model claimed but that don't appear verbatim in the input —
    dropped rather than trusted. A nonzero count here on real traffic is a
    signal worth tracking in the eval, not silently swallowed."""


def classify(message: str) -> ClassifyResult:
    """Run Layer 1 on a message. Raises OllamaUnavailableError if the local
    model isn't running (see app.llm) — never falls back to fabricated
    output or a cloud API."""
    analysis, generation = generate_structured(
        prompt=f"Message:\n{message}",
        schema=Analysis,
        system=_build_system_prompt(),
    )

    kept = []
    dropped = []
    for detection in analysis.detections:
        if detection.span in message:
            kept.append(detection)
        else:
            dropped.append(detection.span)

    return ClassifyResult(
        analysis=Analysis(detections=kept),
        generation=generation,
        dropped_spans=dropped,
    )
