"""Tests for the Layer 1 cascade gate (app/student.py). Pure threshold
logic, mocked at predict_proba_array so no actual model weights need to be
loaded — the model itself is exercised live by eval/compare.py and
eval/gate_calibration.py, not by these unit tests.
"""

from unittest.mock import patch

import numpy as np

from app.student import GATE_THRESHOLD, should_investigate


def test_fires_when_any_technique_clears_threshold():
    proba = np.array([0.01, 0.02, GATE_THRESHOLD, 0.0, 0.0])
    with patch("app.student.predict_proba_array", return_value=proba):
        assert should_investigate("some message") is True


def test_stays_silent_when_nothing_clears_threshold():
    proba = np.zeros(11) + (GATE_THRESHOLD - 0.01)
    with patch("app.student.predict_proba_array", return_value=proba):
        assert should_investigate("some message") is False


def test_custom_threshold_overrides_default():
    proba = np.array([0.5, 0.0, 0.0])
    with patch("app.student.predict_proba_array", return_value=proba):
        assert should_investigate("x", threshold=0.6) is False
        assert should_investigate("x", threshold=0.4) is True


def test_exactly_at_threshold_fires():
    """Boundary check: >= not >, since GATE_THRESHOLD is meant to be
    deliberately permissive (see module docstring — a false positive here
    only costs latency, a false negative skips Layer 2 entirely)."""
    proba = np.array([GATE_THRESHOLD])
    with patch("app.student.predict_proba_array", return_value=proba):
        assert should_investigate("x", threshold=GATE_THRESHOLD) is True
