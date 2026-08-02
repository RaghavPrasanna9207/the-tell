"""Tests for the cost-weighted metrics module — pure numpy logic, no model
or corpus dependency, so correctness here is cheap to pin down exactly.
"""

import numpy as np

from eval.metrics import FN_WEIGHT, FP_WEIGHT, macro_average, per_technique_metrics


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
    from the "FN weighted 5-10x FP" decision in the plan/CLAUDE.md."""
    assert FN_WEIGHT >= 5
    assert FN_WEIGHT <= 10
    assert FP_WEIGHT == 1
