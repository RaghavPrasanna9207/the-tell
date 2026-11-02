"""Integrity tests for taxonomy.py and counter_moves.yaml.

These enforce the non-negotiables from docs/DESIGN_RULES.md: every technique has a
sourced counter-move, no entry is orphaned in either direction, and the
generated JSON schema (fed to the constrained decoder) is well-formed.
"""

from urllib.parse import urlparse

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


# Citations must point at the institutions that actually set the facts this
# product asserts — the cyber-crime authority, the central bank, or the
# government press service. An allowlist rather than a liveness check on
# purpose: these sites block automated requests (pib.gov.in 403s non-browser
# clients, rbi.org.in serves a CAPTCHA), so a HEAD-check test would be flaky
# and CAPTCHA-gated, and a green HEAD only proves a server answered. "Is this
# an authoritative source" is the property worth enforcing, and it's the one
# that stops a plausible-looking blog link sneaking into the fact table.
AUTHORITATIVE_SOURCE_HOSTS = {
    "cybercrime.gov.in",  # National Cyber Crime Reporting Portal / 1930 helpline
    "i4c.mha.gov.in",  # Indian Cyber Crime Coordination Centre
    "www.rbi.org.in",  # Reserve Bank of India
    "rbikehtahai.rbi.org.in",  # RBI public awareness
    "www.npci.org.in",  # National Payments Corporation of India (UPI)
    "www.pib.gov.in",  # Press Information Bureau
}


@pytest.mark.parametrize("technique", list(Technique))
def test_counter_move_has_nonempty_source(technique: Technique):
    move = load_counter_moves()[technique]
    assert move.source.strip(), f"{technique.value} has a blank source"
    assert len(move.source.strip()) > 5, f"{technique.value} source looks like a placeholder"


@pytest.mark.parametrize("technique", list(Technique))
def test_counter_move_source_url_is_followable_and_authoritative(technique: Technique):
    """A citation nobody can follow is barely a citation. Before this existed
    the table carried prose like "I4C advisory on digital arrest scams" with
    no locator, and the only test on it checked that the string was longer
    than 5 characters."""
    move = load_counter_moves()[technique]
    parsed = urlparse(move.source_url)
    assert parsed.scheme == "https", f"{technique.value} source_url must be https, got {move.source_url!r}"
    assert parsed.netloc, f"{technique.value} source_url has no host: {move.source_url!r}"
    assert parsed.netloc in AUTHORITATIVE_SOURCE_HOSTS, (
        f"{technique.value} cites {parsed.netloc!r}, which is not an authoritative source. "
        f"Allowed: {sorted(AUTHORITATIVE_SOURCE_HOSTS)}"
    )


@pytest.mark.parametrize("technique", list(Technique))
def test_counter_move_fields_are_substantive(technique: Technique):
    """Guards against copy-paste placeholder text ('TODO', empty strings)."""
    move = load_counter_moves()[technique]
    for field_name in ("plain_name", "why_it_works", "reality_check", "counter_action"):
        value = getattr(move, field_name)
        assert value.strip(), f"{technique.value}.{field_name} is blank"
        assert "TODO" not in value.upper(), f"{technique.value}.{field_name} still has a TODO"


def test_counter_move_model_rejects_blank_fields():
    # source_url is supplied here so the blank `source` is the ONLY thing
    # wrong — otherwise this passes on the missing-field error instead and
    # stops testing what it says it tests.
    with pytest.raises(ValueError, match="blank"):
        CounterMove(
            id=Technique.ISOLATION,
            plain_name="x",
            why_it_works="x",
            reality_check="x",
            source="   ",  # blank after strip
            source_url="https://cybercrime.gov.in/",
            counter_action="x",
        )


def test_counter_move_model_rejects_blank_source_url():
    with pytest.raises(ValueError, match="blank"):
        CounterMove(
            id=Technique.ISOLATION,
            plain_name="x",
            why_it_works="x",
            reality_check="x",
            source="a real citation",
            source_url="   ",
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
    layer is responsible for never phrasing this as 'safe'. See docs/DESIGN_RULES.md."""
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
