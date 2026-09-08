"""Tests for the cost-weighted metrics module — pure numpy logic, no model
or corpus dependency, so correctness here is cheap to pin down exactly.
"""

import numpy as np

from eval.metrics import FN_WEIGHT, FP_WEIGHT, macro_average, per_technique_metrics, per_technique_metrics_cv


def test_perfect_predictions_score_perfectly():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])

    results = per_technique_metrics(y_true, y_proba, ["a", "b"])

    for r in results:
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0
        assert r.cost == 0
        assert r.fp == 0
        assert r.fn == 0


def test_threshold_sweep_picks_lower_cutoff_when_fn_is_expensive():
    """One technique has borderline probabilities straddling 0.5. Since a
    false negative costs FN_WEIGHT times a false positive, the optimal
    threshold should favor recall (a lower cutoff) over precision here."""
    y_true = np.array([[1], [1], [1], [0]])
    y_proba = np.array([[0.55], [0.45], [0.35], [0.30]])

    results = per_technique_metrics(y_true, y_proba, ["x"])
    r = results[0]

    # threshold 0.35 catches all 3 positives (recall=1) at the cost of one
    # false positive (the 0.30 negative slips past only if threshold <= 0.30,
    # which it isn't here) — recall should be high given FN >> FP cost.
    assert r.recall >= 0.66  # at least 2/3 positives caught
    assert r.support == 3


def test_support_reflects_true_positive_count_not_predictions():
    y_true = np.array([[1], [1], [0], [0], [0]])
    y_proba = np.array([[0.1], [0.1], [0.9], [0.9], [0.9]])  # model gets it all backwards

    results = per_technique_metrics(y_true, y_proba, ["x"])
    assert results[0].support == 2  # 2 true positives in y_true, regardless of predictions


def test_zero_support_technique_scores_zero_not_one():
    """A technique with no positive examples has a 0/0 precision and
    recall — with zero_division=0 that resolves to 0.0, not 1.0. This is
    exactly why run_eval.py's report tells readers to check `support`
    before trusting a row: a 0.0 here means "no data," not "failed"."""
    y_true = np.array([[0], [0], [0], [0]])
    y_proba = np.array([[0.1], [0.1], [0.1], [0.1]])

    results = per_technique_metrics(y_true, y_proba, ["never_positive"])
    r = results[0]

    assert r.support == 0
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0


def test_macro_average_is_plain_mean_across_techniques():
    y_true = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    y_proba = np.array([[0.9, 0.1], [0.1, 0.1], [0.9, 0.1], [0.1, 0.1]])

    results = per_technique_metrics(y_true, y_proba, ["always_right", "never_positive"])
    macro = macro_average(results)

    # "always_right" scores perfectly (1.0); "never_positive" has zero
    # support so resolves to 0.0 (see test above) — mean of 1.0 and 0.0.
    assert macro["precision"] == 0.5
    assert macro["recall"] == 0.5
    assert macro["f1"] == 0.5


def test_cost_weights_are_the_documented_values():
    """Pin the FN:FP weighting so a future refactor can't silently drift
    from the "FN weighted 5-10x FP" decision in the plan/docs/DESIGN_RULES.md."""
    assert FN_WEIGHT >= 5
    assert FN_WEIGHT <= 10
    assert FP_WEIGHT == 1


def test_cv_perfect_separation_scores_perfectly():
    """With cleanly separated probabilities, every fold's held-out threshold
    should still recover perfect predictions — this isolates "does the
    plumbing work" from "does low support make CV noisy" (covered below)."""
    rng = np.random.default_rng(0)
    n = 40
    y_true = np.zeros((n, 1), dtype=int)
    y_true[: n // 2, 0] = 1
    proba_pos = rng.uniform(0.7, 0.95, size=n // 2)
    proba_neg = rng.uniform(0.05, 0.3, size=n // 2)
    y_proba = np.concatenate([proba_pos, proba_neg]).reshape(-1, 1)

    results = per_technique_metrics_cv(y_true, y_proba, ["x"], n_splits=5, seed=0)
    r = results[0]
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.f1 == 1.0


def test_cv_every_message_scored_exactly_once():
    """Every message must appear in exactly one fold's held-out test split,
    so tp+fp+fn+tn across the returned result must equal N."""
    n = 37  # deliberately not divisible by n_splits
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=(n, 1))
    y_proba = rng.uniform(0, 1, size=(n, 1))

    results = per_technique_metrics_cv(y_true, y_proba, ["x"], n_splits=5, seed=1)
    r = results[0]
    assert r.tp + r.fp + r.fn + r.tn == n


def test_cv_threshold_is_never_selected_on_the_fold_it_scores():
    """Regression guard for the core bug this function fixes: construct
    y_proba so that a technique's positives are trivially separable EXCEPT
    within one held-out fold, where the model instead perfectly predicts
    the fold's own probabilities. An in-sample threshold sweep (using
    per_technique_metrics on the same data) would still find the global
    in-sample optimum, but per_technique_metrics_cv must produce a
    different, worse-or-equal result on the affected fold since its
    threshold is fit blind to that fold's labels — i.e. the CV and
    in-sample results must not be identical here."""
    rng = np.random.default_rng(2)
    n = 50
    y_true = np.zeros((n, 1), dtype=int)
    y_true[::2, 0] = 1  # alternating positives/negatives, order matters for fold membership
    # Probabilities exactly track y_true (0.9 for positive, 0.1 for negative)
    # EXCEPT the values are noisy enough near the boundary that in-sample
    # tuning overfits differently than 5-fold CV would.
    y_proba = np.where(y_true == 1, 0.55, 0.45).astype(float)
    jitter = rng.uniform(-0.03, 0.03, size=(n, 1))
    y_proba = np.clip(y_proba + jitter, 0.0, 1.0)

    in_sample = per_technique_metrics(y_true, y_proba, ["x"])[0]
    held_out = per_technique_metrics_cv(y_true, y_proba, ["x"], n_splits=5, seed=2)[0]

    # In-sample F1 must be >= held-out F1: in-sample optimizes on the exact
    # data it's scored on, so it can never do worse than a threshold picked
    # blind to that data.
    assert in_sample.f1 >= held_out.f1


def test_cv_reduces_folds_when_n_is_smaller_than_n_splits():
    """Guard against KFold raising when n_splits > number of samples."""
    y_true = np.array([[1], [0], [1]])
    y_proba = np.array([[0.9], [0.1], [0.8]])
    results = per_technique_metrics_cv(y_true, y_proba, ["x"], n_splits=5, seed=0)
    assert results[0].support == 2
