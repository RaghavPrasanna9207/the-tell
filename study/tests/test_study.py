"""Tests for the study harness's pure logic: block-randomized arm
assignment, input parsing, and the fixed analysis plan's statistics.
Doesn't touch stdin/stdout or the LLM.
"""

import pytest

from analyze_study import bootstrap_diff_ci, is_resistant, wilson_interval
from run_study import next_arm, parse_choice, parse_confidence, parse_yes_no


def test_next_arm_each_block_has_one_of_each():
    arms = [next_arm(i) for i in range(20)]
    for block_start in range(0, 20, 2):
        block = arms[block_start : block_start + 2]
        assert sorted(block) == ["control", "treatment"]


def test_next_arm_is_reproducible():
    assert [next_arm(i) for i in range(10)] == [next_arm(i) for i in range(10)]


def test_parse_choice_valid():
    assert parse_choice("1") == "comply"
    assert parse_choice("5") == "call_someone"


def test_parse_choice_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_choice("0")
    with pytest.raises(ValueError):
        parse_choice("6")


def test_parse_confidence_valid():
    assert parse_confidence("3") == 3


def test_parse_confidence_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_confidence("0")
    with pytest.raises(ValueError):
        parse_confidence("6")


def test_parse_yes_no():
    assert parse_yes_no("y") is True
    assert parse_yes_no("Yes") is True
    assert parse_yes_no("n") is False
    assert parse_yes_no("No") is False
    with pytest.raises(ValueError):
        parse_yes_no("maybe")


def test_is_resistant_mapping():
    assert is_resistant("comply") == 0
    assert is_resistant("reply") == 0
    assert is_resistant("ignore") == 1
    assert is_resistant("verify_independently") == 1
    assert is_resistant("call_someone") == 1


def test_is_resistant_rejects_unknown_code():
    with pytest.raises(ValueError):
        is_resistant("shrug")


def test_wilson_interval_bounds_contain_point_estimate():
    lo, hi = wilson_interval(7, 10)
    assert lo <= 0.7 <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_interval_zero_n_is_degenerate():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_bootstrap_diff_ci_zero_for_identical_arms():
    outcomes = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    point, lo, hi = bootstrap_diff_ci(outcomes, list(outcomes))
    assert point == 0.0
    assert lo <= 0.0 <= hi


def test_bootstrap_diff_ci_detects_large_difference():
    control = [0] * 10
    treatment = [1] * 10
    point, lo, hi = bootstrap_diff_ci(control, treatment)
    assert point == 1.0
    assert lo > 0.0  # CI shouldn't straddle 0 for a maximally stark difference


def test_bootstrap_diff_ci_empty_arm_is_safe():
    assert bootstrap_diff_ci([], [1, 0, 1]) == (0.0, 0.0, 0.0)
