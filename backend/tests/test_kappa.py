"""Tests for eval/kappa.py's pure computation logic — no file I/O, no
dependency on a real second annotator existing yet.
"""

import numpy as np

from eval.kappa import _kappa_or_na, jaccard, per_technique_kappa


def test_perfect_agreement_is_kappa_one():
    col1 = np.array([1, 0, 1, 0, 1])
    col2 = np.array([1, 0, 1, 0, 1])
    assert _kappa_or_na(col1, col2) == 1.0


def test_both_raters_constant_and_agreeing_is_na():
    """Zero positives from both raters — 'no data to disagree on', not
    chance-level (0.0) or perfect (1.0) agreement. This is the case
    sklearn's cohen_kappa_score returns NaN for."""
    col1 = np.array([0, 0, 0, 0])
    col2 = np.array([0, 0, 0, 0])
    assert _kappa_or_na(col1, col2) is None


def test_one_rater_constant_other_varies_is_still_defined():
    """Only the BOTH-constant-and-agreeing case is undefined — one rater
    being constant while the other varies is a real (usually low) kappa,
    not a degenerate case."""
    col1 = np.array([0, 0, 0, 0, 0])
    col2 = np.array([1, 0, 1, 0, 1])
    result = _kappa_or_na(col1, col2)
    assert result is not None


def test_partial_disagreement_is_between_zero_and_one():
    col1 = np.array([1, 0, 1, 0, 1, 0])
    col2 = np.array([1, 1, 0, 0, 1, 1])
    result = _kappa_or_na(col1, col2)
    assert result is not None
    assert 0.0 <= result < 1.0


def test_jaccard_identical_label_sets_is_one():
    assert jaccard(["isolation", "false_authority"], ["false_authority", "isolation"]) == 1.0


def test_jaccard_both_empty_is_one_not_undefined():
    """Both annotators saying 'none' is perfect agreement, not a 0/0
    vacuous case — unlike the per-technique kappa degenerate case, this is
    a real, meaningful agreement (they agreed the message is clean)."""
    assert jaccard([], []) == 1.0


def test_jaccard_disjoint_label_sets_is_zero():
    assert jaccard(["isolation"], ["fake_scarcity"]) == 0.0


def test_jaccard_partial_overlap():
    result = jaccard(["isolation", "false_authority"], ["isolation", "fake_scarcity"])
    assert result == 1 / 3  # intersection={isolation}, union has 3 elements


def test_per_technique_kappa_reports_support_per_technique():
    # 2 techniques, 4 messages: technique 0 has 2 positives from each rater,
    # technique 1 has 0 positives from both (degenerate -> None)
    y1 = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    y2 = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])

    rows = per_technique_kappa(y1, y2, ["fake_scarcity", "isolation"])
    assert len(rows) == 2
    assert rows[0]["kappa"] == 1.0
    assert rows[0]["support_primary"] == 2
    assert rows[0]["support_second"] == 2
    assert rows[1]["kappa"] is None  # zero support from both -> undefined
    assert rows[1]["support_primary"] == 0
