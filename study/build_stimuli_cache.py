"""Precompute Layer 1 + Layer 2 output for the study's six fixed stimuli and
freeze the result to `stimuli_cache.json`.

Why frozen rather than generated live during each study session: every
participant needs to see the EXACT SAME treatment text, but the teacher
model is not run-to-run deterministic even at low temperature. Freezing the
output once — using the real production pipeline, not hand-written text —
removes sampling variance as a confound while keeping the "model selects
and phrases, never invents" guarantee intact: the frozen text is real
Layer 1/2 output, just computed once instead of live per participant.

Requires Ollama running locally with the teacher model pulled. Re-run this
only if the stimuli text or the classify/explain pipeline changes.

Run: python study/build_stimuli_cache.py
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.classify import classify  # noqa: E402
from app.explain import explain_all  # noqa: E402

from stimuli import STIMULI  # noqa: E402

CACHE_PATH = Path(__file__).parent / "stimuli_cache.json"


def build_entry(stimulus: dict) -> dict:
    classify_result = classify(stimulus["text"])
    is_clean = classify_result.analysis.is_clean

    cards = []
    if not is_clean:
        for result in explain_all(classify_result.analysis.detections):
            cards.append(
                {
                    "technique": result.detection.technique.value,
                    "plain_name": result.counter_move.plain_name,
                    "span": result.detection.span,
                    "confidence": result.detection.confidence,
                    "explanation": result.explanation,
                    "source": result.counter_move.source,
                }
            )

    return {
        "id": stimulus["id"],
        "kind": stimulus["kind"],
        "text": stimulus["text"],
        "is_clean": is_clean,
        "cards": cards,
    }


def main() -> None:
    entries = [build_entry(s) for s in STIMULI]
    CACHE_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Cached {len(entries)} stimuli to {CACHE_PATH}")
    for e in entries:
        print(f"  {e['id']}: is_clean={e['is_clean']} cards={len(e['cards'])}")


if __name__ == "__main__":
    main()
