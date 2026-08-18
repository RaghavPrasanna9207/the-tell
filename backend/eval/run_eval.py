"""Eval harness entry point: the TF-IDF + LogisticRegression baseline only,
via cross-validation against the real gold set. For the full baseline /
teacher / student three-way comparison (per-technique + macro + latency +
model size), see `eval/compare.py` — this script is the quick baseline-only
check.

Run: python eval/run_eval.py
"""

import sys
from pathlib import Path

# So `python eval/run_eval.py` works regardless of cwd, matching the plan's
# documented invocation — without this, `eval` and `app` aren't resolvable
# as packages when the script is run directly rather than via `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.baseline import load_gold_as_arrays, run_baseline_cv
from eval.gold import GOLD_NOTE, load_real_gold
from eval.metrics import format_report, per_technique_metrics, per_technique_metrics_cv


def main() -> None:
    records = load_real_gold()
    n_positive = sum(1 for r in records if r["labels"])
    n_clean = sum(1 for r in records if not r["labels"])

    print("=" * 70)
    print(GOLD_NOTE)
    print(f"   N = {len(records)} ({n_positive} scam, {n_clean} clean)")
    print("=" * 70)

    texts, y_true, technique_names = load_gold_as_arrays(records)
    y_proba = run_baseline_cv(texts, y_true)

    in_sample = per_technique_metrics(y_true, y_proba, technique_names)
    print(
        format_report(
            in_sample,
            "BASELINE: TF-IDF + Logistic Regression (5-fold CV probabilities, "
            "IN-SAMPLE threshold - upper bound, see eval/compare.py for held-out)",
        )
    )

    cv = per_technique_metrics_cv(y_true, y_proba, technique_names)
    print(
        format_report(
            cv,
            "BASELINE: TF-IDF + Logistic Regression (5-fold CV probabilities, "
            "HELD-OUT 5-fold threshold CV - genuine generalization estimate)",
        )
    )

    print(
        "\nNo positive examples for a technique in the eval set means its P/R/F1 "
        "are vacuous (0/0) - check `support` before trusting any row."
    )


if __name__ == "__main__":
    main()
