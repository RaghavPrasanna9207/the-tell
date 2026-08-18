"""Cohen's kappa between two annotators on the gold-labeling kappa subset
(see eval/label.py's --kappa-subset and the plan's Week 1 step 5 /
Phase A4: "if kappa < 0.6 the taxonomy is ambiguous — fix the taxonomy,
don't fudge the number").

Multi-label agreement is reported two ways, because they answer different
questions:

- Per-technique kappa (11 rows): treats each technique as an independent
  binary yes/no rater decision. This is what actually tests taxonomy
  ambiguity — two annotators can agree "something's wrong here" while
  disagreeing on WHICH technique (e.g. the documented false_authority vs
  trust_transfer mix-up), and only the per-technique breakdown catches
  that. Several techniques will have very low support even on the intended
  80-message subset (they already do on the full 250, see
  ERROR_ANALYSIS.md) — kappa on a handful of positives is close to
  statistical noise, so support is reported alongside every row and
  near-zero-support rows are flagged, not silently averaged in as if they
  were trustworthy.
- Message-level "any technique present" binary kappa, exact-match rate, and
  mean Jaccard similarity: coarser, well-powered (all N messages
  contribute, not a per-technique sliver of them) sanity checks that don't
  depend on getting the specific technique right.

Compares whatever ids exist in BOTH labels_primary.jsonl and
labels_second.jsonl (the intersection), so this also works mid-labeling —
no need to have all 80 done to get a running number, though the
plan's target is the full 80.

Run: python eval/kappa.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MultiLabelBinarizer

from app.taxonomy import Technique
from eval.label import LABELS_DIR

PRIMARY_PATH = LABELS_DIR / "labels_primary.jsonl"
SECOND_PATH = LABELS_DIR / "labels_second.jsonl"
KAPPA_SUBSET_SIZE = 80  # matches the plan's "kappa on an 80-message subset"

TECHNIQUE_NAMES = [t.value for t in Technique]


def _load_labels(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                records[entry["id"]] = {"text": entry["text"], "labels": entry["labels"]}
    return records


def _kappa_or_na(col1: np.ndarray, col2: np.ndarray) -> float | None:
    """None means undefined (both raters are constant on the same single
    value across every message — zero positives from both, or, in
    principle, both always-positive). sklearn's cohen_kappa_score would
    return NaN there (0/0 in the kappa formula's denominator) — reporting
    that as a number, or worse coercing it to 0.0 or 1.0, would misrepresent
    "no data to disagree on" as either "chance-level agreement" or "perfect
    agreement", neither of which is true."""
    if len(set(col1.tolist())) == 1 and len(set(col2.tolist())) == 1 and col1[0] == col2[0]:
        return None
    return float(cohen_kappa_score(col1, col2))


def per_technique_kappa(y1: np.ndarray, y2: np.ndarray, technique_names: list[str]) -> list[dict]:
    rows = []
    for i, name in enumerate(technique_names):
        col1, col2 = y1[:, i], y2[:, i]
        rows.append(
            {
                "technique": name,
                "kappa": _kappa_or_na(col1, col2),
                "support_primary": int(col1.sum()),
                "support_second": int(col2.sum()),
            }
        )
    return rows


def jaccard(labels1: list[str], labels2: list[str]) -> float:
    s1, s2 = set(labels1), set(labels2)
    if not s1 and not s2:
        return 1.0  # both said "none" — perfect agreement, not vacuous
    return len(s1 & s2) / len(s1 | s2)


def main() -> None:
    if not PRIMARY_PATH.exists():
        print(f"ERROR: {PRIMARY_PATH} not found.")
        sys.exit(1)
    if not SECOND_PATH.exists():
        print(
            f"ERROR: {SECOND_PATH} not found. Get a second annotator to label the kappa "
            f"subset first — either:\n"
            f"  python eval/label.py --annotator second --kappa-subset {KAPPA_SUBSET_SIZE}\n"
            f"or, for a friendlier spreadsheet handoff:\n"
            f"  python eval/export_labeling_sheet.py --annotator second --kappa-subset {KAPPA_SUBSET_SIZE}\n"
            f"  (send them gold_labeling_second.xlsx, they fill it in, you get it back)\n"
            f"  python eval/import_labeling_sheet.py --annotator second"
        )
        sys.exit(1)

    primary = _load_labels(PRIMARY_PATH)
    second = _load_labels(SECOND_PATH)

    shared_ids = sorted(set(primary) & set(second))
    if not shared_ids:
        print("ERROR: no overlapping message ids between labels_primary.jsonl and labels_second.jsonl.")
        sys.exit(1)
    if len(shared_ids) < KAPPA_SUBSET_SIZE:
        print(
            f"WARNING: only {len(shared_ids)}/{KAPPA_SUBSET_SIZE} of the intended kappa subset "
            f"are labeled by both annotators so far — numbers below are a running total, not final."
        )

    print("=" * 70)
    print(f"Cohen's kappa: labels_primary.jsonl vs labels_second.jsonl, N = {len(shared_ids)}")
    print("=" * 70)

    mlb = MultiLabelBinarizer(classes=TECHNIQUE_NAMES)
    y1 = mlb.fit_transform([primary[i]["labels"] for i in shared_ids])
    y2 = mlb.fit_transform([second[i]["labels"] for i in shared_ids])

    rows = per_technique_kappa(y1, y2, TECHNIQUE_NAMES)
    print(f"\n{'technique':<28} {'kappa':>8} {'support(p)':>11} {'support(s)':>11}")
    defined = []
    for r in rows:
        kappa_str = f"{r['kappa']:.2f}" if r["kappa"] is not None else "N/A"
        flag = "  <- support too low to trust" if r["kappa"] is not None and min(r["support_primary"], r["support_second"]) < 5 else ""
        print(f"{r['technique']:<28} {kappa_str:>8} {r['support_primary']:>11} {r['support_second']:>11}{flag}")
        if r["kappa"] is not None:
            defined.append(r["kappa"])

    if defined:
        macro = sum(defined) / len(defined)
        print(f"\nMacro-average kappa (over {len(defined)}/{len(rows)} techniques with a defined value): {macro:.2f}")
        if macro < 0.6:
            print("Below 0.6 — per the plan, treat this as a taxonomy-ambiguity signal, not a labeling-quality problem.")
    else:
        print("\nNo technique had a defined kappa (every one was constant-and-agreeing in this subset).")

    any_technique_1 = (y1.sum(axis=1) > 0).astype(int)
    any_technique_2 = (y2.sum(axis=1) > 0).astype(int)
    message_kappa = _kappa_or_na(any_technique_1, any_technique_2)
    exact_match = np.mean(
        [set(primary[i]["labels"]) == set(second[i]["labels"]) for i in shared_ids]
    )
    mean_jaccard = np.mean([jaccard(primary[i]["labels"], second[i]["labels"]) for i in shared_ids])

    print("\nMessage-level agreement (well-powered — uses all N messages, not a per-technique sliver):")
    print(f"  'any technique present' binary kappa: {message_kappa:.2f}" if message_kappa is not None else "  'any technique present' binary kappa: N/A")
    print(f"  exact label-set match rate:           {exact_match:.2f}")
    print(f"  mean Jaccard similarity:               {mean_jaccard:.2f}")

    disagreements = [i for i in shared_ids if set(primary[i]["labels"]) != set(second[i]["labels"])]
    print(f"\n{len(disagreements)}/{len(shared_ids)} messages have a label-set disagreement:")
    for i in disagreements:
        print(f"\n  [{i}] {primary[i]['text'][:80]!r}")
        print(f"    primary: {sorted(primary[i]['labels']) or ['none']}")
        print(f"    second:  {sorted(second[i]['labels']) or ['none']}")


if __name__ == "__main__":
    main()
