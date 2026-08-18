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
from sklearn.model_selection import KFold

FN_WEIGHT = 8  # false negative: missed a real manipulation technique
FP_WEIGHT = 1  # false positive: flagged a technique that wasn't there

THRESHOLD_GRID = np.arange(0.05, 1.0, 0.05)
CALIBRATION_FOLDS = 5


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


def _select_threshold(y_true_col: np.ndarray, proba_col: np.ndarray) -> float:
    """Sweep THRESHOLD_GRID and return the threshold minimizing cost-weighted
    error on the given (y_true_col, proba_col). Note this is agnostic to
    whether the caller passes in-sample or held-out data — it's the
    "held-out" property of *what gets passed in* that matters, not this
    function.
    """
    best_threshold = 0.5
    best_cost = float("inf")
    for threshold in THRESHOLD_GRID:
        y_pred_col = (proba_col >= threshold).astype(int)
        cost = _cost(y_true_col, y_pred_col)
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold


def _score_at_threshold(y_true_col: np.ndarray, y_pred_col: np.ndarray, threshold: float, name: str) -> TechniqueResult:
    tp = int(np.sum((y_true_col == 1) & (y_pred_col == 1)))
    fp = int(np.sum((y_true_col == 0) & (y_pred_col == 1)))
    fn = int(np.sum((y_true_col == 1) & (y_pred_col == 0)))
    tn = int(np.sum((y_true_col == 0) & (y_pred_col == 0)))
    return TechniqueResult(
        technique=name,
        threshold=threshold,
        precision=precision_score(y_true_col, y_pred_col, zero_division=0),
        recall=recall_score(y_true_col, y_pred_col, zero_division=0),
        f1=f1_score(y_true_col, y_pred_col, zero_division=0),
        cost=_cost(y_true_col, y_pred_col),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        support=int(np.sum(y_true_col)),
    )


def per_technique_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, technique_names: list[str]
) -> list[TechniqueResult]:
    """For each technique (column), sweep THRESHOLD_GRID and pick the
    threshold that minimizes cost-weighted error, then report P/R/F1 at
    that threshold — evaluated on the SAME data the threshold was chosen
    from. This is in-sample threshold selection: every number here is an
    upper bound on what a threshold picked without seeing these labels
    would achieve, especially for techniques with only a handful of
    positives. See `per_technique_metrics_cv` for the held-out variant, and
    ERROR_ANALYSIS.md for why this matters here.
    """
    results = []
    for i, name in enumerate(technique_names):
        y_true_col = y_true[:, i]
        proba_col = y_proba[:, i]
        threshold = _select_threshold(y_true_col, proba_col)
        y_pred_col = (proba_col >= threshold).astype(int)
        results.append(_score_at_threshold(y_true_col, y_pred_col, threshold, name))
    return results


def per_technique_metrics_cv(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    technique_names: list[str],
    n_splits: int = CALIBRATION_FOLDS,
    seed: int = 42,
) -> list[TechniqueResult]:
    """Held-out threshold selection via K-fold: for each fold, the
    per-technique threshold is chosen by cost-minimization on the OTHER
    folds only, then applied to score this fold. Concatenating every fold's
    held-out predictions gives every message a prediction from a threshold
    that never saw its own label — a genuine generalization estimate,
    unlike `per_technique_metrics`.

    K-fold rather than a single calibration/test split: several techniques
    have single-digit gold support (as low as 3), so a static split could
    leave a technique with 0-1 positives in the test half, making that
    row's F1 a coin flip on the split's luck rather than a real signal.
    K-fold uses every message for both calibration and (held-out)
    evaluation, at the cost of a `threshold` column that's now a mean
    across folds rather than one fixed value — low-support techniques will
    still show real fold-to-fold instability; that instability is the
    honest finding, not a bug in this function.

    Does not retrain any model - y_proba is already-computed probabilities
    (from a live teacher call or a single student forward pass); this only
    cross-validates the threshold-selection step against them.
    """
    n = y_true.shape[0]
    kf = KFold(n_splits=min(n_splits, n), shuffle=True, random_state=seed)
    fold_indices = list(kf.split(np.arange(n)))

    results = []
    for i, name in enumerate(technique_names):
        y_true_col = y_true[:, i]
        proba_col = y_proba[:, i]
        y_pred_col = np.zeros(n, dtype=int)
        fold_thresholds = []
        for train_idx, test_idx in fold_indices:
            threshold = _select_threshold(y_true_col[train_idx], proba_col[train_idx])
            fold_thresholds.append(threshold)
            y_pred_col[test_idx] = (proba_col[test_idx] >= threshold).astype(int)
        mean_threshold = float(np.mean(fold_thresholds))
        results.append(_score_at_threshold(y_true_col, y_pred_col, mean_threshold, name))
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
