"""Tests for the Layer 2 validator and fallback path. These are pure
functions with no Ollama dependency — they must pass without the LLM
running, since they encode the safety guarantees the LLM is not trusted to
provide on its own.
"""

from app.counter_moves import load_counter_moves
from app.explain import (
    MAX_PURPOSE_WORDS,
    _fallback_explanation,
    validate_purpose_sentence,
)
from app.taxonomy import Detection, Technique


def test_valid_sentence_passes():
    span = "do not tell your family"
    sentence = f'The instruction "{span}" exists to stop anyone from calling you before you send money.'
    assert validate_purpose_sentence(sentence, span) == []


def test_missing_span_fails():
    problems = validate_purpose_sentence("This creates urgency to rush you.", "within 2 hours")
    assert any("verbatim span" in p for p in problems)


def test_too_long_fails():
    span = "act now"
    sentence = f'"{span}" ' + "word " * (MAX_PURPOSE_WORDS + 5)
    problems = validate_purpose_sentence(sentence, span)
    assert any("too long" in p for p in problems)


def test_hedging_phrase_fails():
    span = "CBI officer"
    sentence = f'"{span}" might be a scam and you should be careful.'
    problems = validate_purpose_sentence(sentence, span)
    assert any("hedging" in p for p in problems)


def test_valid_sentence_at_word_boundary_passes():
    span = "urgent"
    words = ["word"] * (MAX_PURPOSE_WORDS - 1) + [f'"{span}"']
    sentence = " ".join(words)
    assert validate_purpose_sentence(sentence, span) == []


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
