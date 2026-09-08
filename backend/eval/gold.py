"""Loader for the hand-labeled evaluation set.

Two loaders, and it matters which one a caller uses:

`load_real_gold()` — the 250-message hand-labeled set
(`data/corpus/gold/labels_primary.jsonl`). This is what every reported number
should be scored against.

`load_draft_gold()` — `draft_labels` from
`data/corpus/raw/handcrafted_india.jsonl`, technique ids assigned by the same
person who wrote the messages, at authoring time. A pilot, kept only as a
smoke-test fixture. Anything reported from it carries DRAFT_GOLD_WARNING.

HONESTY NOTE (see docs/DESIGN_RULES.md "report metrics honestly"): the real
gold set is single-annotator, with macro-average Cohen's kappa 0.68 against a
second annotator on an 80-message subset. Its composition limits what any
score against it means — see GOLD_NOTE below.
"""

import json
from pathlib import Path

from app.taxonomy import Technique

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus" / "raw" / "handcrafted_india.jsonl"
LABELS_PATH = Path(__file__).parent.parent / "data" / "corpus" / "gold" / "labels_primary.jsonl"

DRAFT_GOLD_WARNING = (
    "DRAFT LABELS, NOT VERIFIED GOLD - author-assigned at message-authoring "
    "time, no second annotator, no kappa computed. N is far below the "
    "planned ~250-message hand-labeled set. Treat these numbers as a rough "
    "pilot, not a result."
)

GOLD_NOTE = (
    "REAL GOLD SET - N=250, every message individually reviewed (see "
    "data/corpus/gold/labels_primary.jsonl). Labels are single-annotator; a "
    "second annotator independently labeled an 80-message subset, giving "
    "macro-average Cohen's kappa = 0.68 (eval/kappa.py, ERROR_ANALYSIS.md). "
    "Composition matters when reading any number scored against this set: "
    "190 of 250 messages are UCI SMS spam and 45 of the 69 positive examples "
    "are author-written, so this measures the taxonomy against its author's "
    "own distribution more than against live Indian scam traffic."
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


def load_real_gold() -> list[dict]:
    """Load the real hand-labeled gold set: all 250 sampled messages from
    `data/corpus/gold/labels_primary.jsonl`, every one individually
    reviewed. Same shape and validation guarantee as `load_draft_gold`.
    """
    valid = {t.value for t in Technique}
    records = []
    with LABELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            labels = entry.get("labels") or []
            for label in labels:
                if label not in valid:
                    raise ValueError(f"{entry['id']} has invalid label {label!r}")
            records.append({"id": entry["id"], "text": entry["text"], "labels": labels})
    return records
