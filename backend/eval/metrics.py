"""Cost-weighted multi-label classification metrics.

A missed manipulation technique (false negative) is worse than a
false-alarm (false positive) — a false negative means the app stayed
silent on a real scam. FN_WEIGHT / FP_WEIGHT encode that explicitly rather
than optimizing plain accuracy or F1, which treats both errors as equally
bad. The operating threshold per technique is chosen to minimize this
weighted cost, not to maximize F1 — that's the whole point of a
cost-weighted eval instead of a single accuracy number.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

FN_WEIGHT = 8  # false negative: missed a real manipulation technique
FP_WEIGHT = 1  # false positive: flagged a technique that wasn't there

THRESHOLD_GRID = np.arange(0.05, 1.0, 0.05)


@dataclass
class TechniqueResult:
    technique: str
    threshold: float
    precision: float
    recall: float
    f1: float
    cost: float
    tp: int
    fp: int
    fn: int
    tn: int
    support: int
    """Number of positive examples for this technique in y_true."""


def _cost(y_true_col: np.ndarray, y_pred_col: np.ndarray) -> float:
    fn = int(np.sum((y_true_col == 1) & (y_pred_col == 0)))
    fp = int(np.sum((y_true_col == 0) & (y_pred_col == 1)))
    return FN_WEIGHT * fn + FP_WEIGHT * fp


def per_technique_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, technique_names: list[str]
) -> list[TechniqueResult]:
    """For each technique (column), sweep THRESHOLD_GRID and pick the
    threshold that minimizes cost-weighted error, then report P/R/F1 at
    that threshold.
    """
    results = []
    for i, name in enumerate(technique_names):
        y_true_col = y_true[:, i]
        proba_col = y_proba[:, i]

        best_threshold = 0.5
        best_cost = float("inf")
        for threshold in THRESHOLD_GRID:
            y_pred_col = (proba_col >= threshold).astype(int)
            cost = _cost(y_true_col, y_pred_col)
            if cost < best_cost:
                best_cost = cost
                best_threshold = float(threshold)

        y_pred_col = (proba_col >= best_threshold).astype(int)
        tp = int(np.sum((y_true_col == 1) & (y_pred_col == 1)))
        fp = int(np.sum((y_true_col == 0) & (y_pred_col == 1)))
        fn = int(np.sum((y_true_col == 1) & (y_pred_col == 0)))
        tn = int(np.sum((y_true_col == 0) & (y_pred_col == 0)))

        results.append(
            TechniqueResult(
                technique=name,
                threshold=best_threshold,
                precision=precision_score(y_true_col, y_pred_col, zero_division=0),
                recall=recall_score(y_true_col, y_pred_col, zero_division=0),
                f1=f1_score(y_true_col, y_pred_col, zero_division=0),
                cost=best_cost,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                support=int(np.sum(y_true_col)),
            )
        )
    return results


def macro_average(results: list[TechniqueResult]) -> dict:
    return {
        "precision": float(np.mean([r.precision for r in results])),
        "recall": float(np.mean([r.recall for r in results])),
        "f1": float(np.mean([r.f1 for r in results])),
        "total_cost": float(np.sum([r.cost for r in results])),
    }


def format_report(results: list[TechniqueResult], title: str) -> str:
    lines = [f"\n=== {title} ===", f"(FN weighted {FN_WEIGHT}x FP; threshold chosen to minimize cost)\n"]
    lines.append(f"{'technique':<28} {'support':>7} {'thr':>5} {'P':>6} {'R':>6} {'F1':>6} {'cost':>6}")
    for r in sorted(results, key=lambda r: r.technique):
        lines.append(
            f"{r.technique:<28} {r.support:>7} {r.threshold:>5.2f} "
            f"{r.precision:>6.2f} {r.recall:>6.2f} {r.f1:>6.2f} {r.cost:>6.0f}"
        )
    macro = macro_average(results)
    lines.append("-" * 70)
    lines.append(
        f"{'MACRO AVERAGE':<28} {'':>7} {'':>5} "
        f"{macro['precision']:>6.2f} {macro['recall']:>6.2f} {macro['f1']:>6.2f} {macro['total_cost']:>6.0f}"
    )
    return "\n".join(lines)
