"""Loader for the hand-labeled evaluation set.

IMPORTANT — HONESTY NOTE (see CLAUDE.md "report metrics honestly"):
The only labeled data that exists right now is `draft_labels` from
`data/corpus/raw/handcrafted_india.jsonl` — technique ids assigned by the
same person who wrote the messages, at authoring time. This is NOT the
~250-message hand-labeled gold set with inter-annotator agreement that the
plan calls for; it's a small pilot (N≈45 scam examples) that exists so the
eval harness has something real to run against today. Every report this
loader feeds into must say so, not present these numbers as final.
"""

import json
from pathlib import Path

from app.taxonomy import Technique

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus" / "raw" / "handcrafted_india.jsonl"

DRAFT_GOLD_WARNING = (
    "DRAFT LABELS, NOT VERIFIED GOLD - author-assigned at message-authoring "
    "time, no second annotator, no kappa computed. N is far below the "
    "planned ~250-message hand-labeled set. Treat these numbers as a rough "
    "pilot, not a result."
)


def load_draft_gold() -> list[dict]:
    """Load the hand-crafted India corpus with its draft technique labels.

    Returns a list of {"id", "text", "labels": list[str]} — every label is
    guaranteed to be a valid Technique value (raises otherwise, so a typo'd
    label fails loudly rather than silently vanishing from the eval).
    """
    valid = {t.value for t in Technique}
    records = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            labels = entry.get("draft_labels") or []
            for label in labels:
                if label not in valid:
                    raise ValueError(f"{entry['id']} has invalid draft_label {label!r}")
            records.append({"id": entry["id"], "text": entry["text"], "labels": labels})
    return records
