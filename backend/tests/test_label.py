"""Tests for the labeling CLI's pure logic — sampling reproducibility and
input parsing. Doesn't touch stdin/stdout, so no need to mock the
interactive loop itself.
"""

import pytest

from eval.label import build_sample, parse_input


def test_parse_input_none():
    assert parse_input("0") == []


def test_parse_input_skip():
    assert parse_input("s") is None


def test_parse_input_single_technique():
    labels = parse_input("1")
    assert len(labels) == 1


def test_parse_input_multiple_techniques():
    labels = parse_input("1,3,5")
    assert len(labels) == 3
    assert len(set(labels)) == 3  # no duplicates from the input itself


def test_parse_input_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_input("999")
    with pytest.raises(ValueError):
        parse_input("0,5")  # mixing "none" with a real label doesn't make sense
    # ^ actually 0 is only special-cased alone; "0,5" hits the general parser
    # and 0 is out of the valid 1-11 range there, which is the correct
    # rejection (ambiguous input, not a silent partial-accept)


def test_build_sample_is_reproducible_with_same_seed():
    """This is the whole point of --seed: two annotators must get the
    exact same message set, or Cohen's kappa is comparing different data."""
    records = [{"id": f"r{i}", "text": f"message {i}", "source": "uci_sms_spam"} for i in range(500)]

    sample_a = build_sample(records, target=100, seed=42)
    sample_b = build_sample(records, target=100, seed=42)

    ids_a = [r["id"] for r in sample_a]
    ids_b = [r["id"] for r in sample_b]
    assert ids_a == ids_b


def test_build_sample_different_seeds_differ():
    records = [{"id": f"r{i}", "text": f"message {i}", "source": "uci_sms_spam"} for i in range(500)]

    sample_a = build_sample(records, target=100, seed=42)
    sample_b = build_sample(records, target=100, seed=7)

    ids_a = [r["id"] for r in sample_a]
    ids_b = [r["id"] for r in sample_b]
    assert ids_a != ids_b


def test_build_sample_always_includes_all_handcrafted():
    """The India-specific messages are the ones that actually exercise the
    taxonomy — they must never be dropped by the random sampling."""
    handcrafted = [
        {"id": f"IN-{i:03d}", "text": f"scam {i}", "source": "handcrafted_india"} for i in range(60)
    ]
    uci = [{"id": f"uci-{i:05d}", "text": f"msg {i}", "source": "uci_sms_spam"} for i in range(1000)]
    records = handcrafted + uci

    sample = build_sample(records, target=250, seed=42)
    sample_ids = {r["id"] for r in sample}
    handcrafted_ids = {r["id"] for r in handcrafted}

    assert handcrafted_ids <= sample_ids


def test_build_sample_respects_target_size():
    handcrafted = [{"id": f"IN-{i:03d}", "text": f"x{i}", "source": "handcrafted_india"} for i in range(60)]
    uci = [{"id": f"uci-{i:05d}", "text": f"y{i}", "source": "uci_sms_spam"} for i in range(1000)]
    records = handcrafted + uci

    sample = build_sample(records, target=250, seed=42)
    assert len(sample) == 250
