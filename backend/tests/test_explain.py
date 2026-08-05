"""Tests for the Layer 2 validator and fallback path. These are pure
functions with no Ollama dependency — they must pass without the LLM
running, since they encode the safety guarantees the LLM is not trusted to
provide on its own.
"""

from app.counter_moves import load_counter_moves
from app.explain import (
    MAX_PURPOSE_WORDS,
    _fallback_explanation,
    validate_purpose_clause,
)
from app.taxonomy import Detection, Technique


def test_valid_purpose_passes():
    span = "do not tell your family"
    purpose = "This exists to stop anyone from calling you before you send money."
    assert validate_purpose_clause(purpose, span) == []


def test_purpose_repeating_span_fails():
    """A real failure seen in practice: the model just echoes the span back
    instead of explaining why it's being used."""
    span = "Only 3 winners have been selected today"
    purpose = f"{span} which creates false scarcity to rush your decision."
    problems = validate_purpose_clause(purpose, span)
    assert any("repeats the verbatim span" in p for p in problems)


def test_too_long_fails():
    span = "act now"
    purpose = "word " * (MAX_PURPOSE_WORDS + 5)
    problems = validate_purpose_clause(purpose, span)
    assert any("too long" in p for p in problems)


def test_hedging_phrase_fails():
    span = "CBI officer"
    purpose = "This might be a scam and you should be careful."
    problems = validate_purpose_clause(purpose, span)
    assert any("hedging" in p for p in problems)


def test_valid_purpose_at_word_boundary_passes():
    words = ["word"] * MAX_PURPOSE_WORDS
    purpose = " ".join(words)
    assert validate_purpose_clause(purpose, "urgent") == []


def test_fallback_explanation_is_table_only_and_grounded():
    """The fallback must never touch the LLM — everything in it comes from
    the detection's span (already verified in Layer 1) and the counter-move
    table (source-cited)."""
    moves = load_counter_moves()
    detection = Detection(technique=Technique.ISOLATION, span="don't tell your family", confidence=0.9)
    counter_move = moves[Technique.ISOLATION]

    text = _fallback_explanation(detection, counter_move)

    assert detection.span in text
    assert counter_move.reality_check.strip() in text
    assert counter_move.counter_action.strip() in text


def test_fallback_never_hedges():
    moves = load_counter_moves()
    for technique, counter_move in moves.items():
        detection = Detection(technique=technique, span="test span", confidence=0.5)
        text = _fallback_explanation(detection, counter_move).lower()
        for phrase in ("might be a scam", "could be a scam", "possibly a scam"):
            assert phrase not in text, f"{technique.value} fallback hedges"
