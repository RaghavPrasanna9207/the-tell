"""Eval harness entry point.

Currently runs the TF-IDF + LogisticRegression baseline via cross-validation
against the draft-labeled corpus. The Ollama teacher and distilled student
columns get added once corpus labeling (Batch-equivalent local labeling)
and distillation land — see the plan's Week 2.

Run: python eval/run_eval.py
"""

import sys
from pathlib import Path

# So `python eval/run_eval.py` works regardless of cwd, matching the plan's
# documented invocation — without this, `eval` and `app` aren't resolvable
# as packages when the script is run directly rather than via `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.baseline import load_gold_as_arrays, run_baseline_cv
from eval.gold import DRAFT_GOLD_WARNING, load_draft_gold
from eval.metrics import format_report, per_technique_metrics


def main() -> None:
    records = load_draft_gold()
    n_positive = sum(1 for r in records if r["labels"])
    n_clean = sum(1 for r in records if not r["labels"])

    print("=" * 70)
    print(f"WARNING: {DRAFT_GOLD_WARNING}")
    print(f"   N = {len(records)} ({n_positive} scam, {n_clean} clean)")
    print("=" * 70)

    texts, y_true, technique_names = load_gold_as_arrays(records)
    y_proba = run_baseline_cv(texts, y_true)

    results = per_technique_metrics(y_true, y_proba, technique_names)
    print(format_report(results, "BASELINE: TF-IDF + Logistic Regression (5-fold CV, out-of-fold predictions)"))

    print(
        "\nNo positive examples for a technique in the eval set means its P/R/F1 "
        "are vacuous (0/0) - check `support` before trusting any row."
    )


if __name__ == "__main__":
    main()
