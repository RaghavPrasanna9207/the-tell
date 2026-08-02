"""Integrity tests for taxonomy.py and counter_moves.yaml.

These enforce the non-negotiables from CLAUDE.md: every technique has a
sourced counter-move, no entry is orphaned in either direction, and the
generated JSON schema (fed to the constrained decoder) is well-formed.
"""

import pytest

from app.counter_moves import CounterMove, load_counter_moves
from app.taxonomy import (
    ALL_TECHNIQUES,
    TECHNIQUE_DESCRIPTIONS,
    Analysis,
    Detection,
    Technique,
    analysis_json_schema,
)


def test_taxonomy_has_eleven_techniques():
    assert len(ALL_TECHNIQUES) == 11
    assert len(set(ALL_TECHNIQUES)) == 11  # no duplicates


def test_every_technique_has_a_runtime_description():
    """Enum member docstrings in taxonomy.py are decorative only — Python
    doesn't attach them at runtime — so prompt-building must read from
    TECHNIQUE_DESCRIPTIONS instead. Guard against the two drifting apart."""
    for technique in Technique:
        assert technique in TECHNIQUE_DESCRIPTIONS, f"missing runtime description for {technique.value}"
        assert TECHNIQUE_DESCRIPTIONS[technique].strip()


def test_every_technique_has_a_counter_move():
    moves = load_counter_moves()
    for technique in Technique:
        assert technique in moves, f"missing counter-move for {technique.value}"


def test_no_orphan_counter_moves():
    """Every entry in the YAML must map to a real taxonomy technique —
    load_counter_moves() would already raise on an unparseable id, but this
    guards against silent drift if the taxonomy shrinks."""
    moves = load_counter_moves()
    assert set(moves.keys()) <= set(Technique)
    assert set(moves.keys()) == set(Technique)


@pytest.mark.parametrize("technique", list(Technique))
def test_counter_move_has_nonempty_source(technique: Technique):
    move = load_counter_moves()[technique]
    assert move.source.strip(), f"{technique.value} has a blank source"
    assert len(move.source.strip()) > 5, f"{technique.value} source looks like a placeholder"


@pytest.mark.parametrize("technique", list(Technique))
def test_counter_move_fields_are_substantive(technique: Technique):
    """Guards against copy-paste placeholder text ('TODO', empty strings)."""
    move = load_counter_moves()[technique]
    for field_name in ("plain_name", "why_it_works", "reality_check", "counter_action"):
        value = getattr(move, field_name)
        assert value.strip(), f"{technique.value}.{field_name} is blank"
        assert "TODO" not in value.upper(), f"{technique.value}.{field_name} still has a TODO"


def test_counter_move_model_rejects_blank_fields():
    with pytest.raises(ValueError):
        CounterMove(
            id=Technique.ISOLATION,
            plain_name="x",
            why_it_works="x",
            reality_check="x",
            source="   ",  # blank after strip
            counter_action="x",
        )


def test_analysis_json_schema_is_valid_and_references_all_techniques():
    schema = analysis_json_schema()
    assert schema["type"] == "object"
    assert "detections" in schema["properties"]

    technique_enum = schema["$defs"]["Technique"]["enum"]
    assert set(technique_enum) == {t.value for t in Technique}


def test_empty_analysis_is_clean_not_unsafe():
    """`is_clean` must exist and mean 'no techniques detected' — the API
    layer is responsible for never phrasing this as 'safe'. See CLAUDE.md."""
    analysis = Analysis()
    assert analysis.is_clean is True
    assert analysis.detections == []


def test_analysis_with_detection_is_not_clean():
    analysis = Analysis(
        detections=[Detection(technique=Technique.ISOLATION, span="don't tell your family", confidence=0.9)]
    )
    assert analysis.is_clean is False
    assert Technique.ISOLATION in analysis.technique_ids


def test_detection_confidence_bounds():
    with pytest.raises(ValueError):
        Detection(technique=Technique.ISOLATION, span="x", confidence=1.5)
    with pytest.raises(ValueError):
        Detection(technique=Technique.ISOLATION, span="x", confidence=-0.1)
