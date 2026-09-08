"""Loader for the grounded fact table (`data/counter_moves.yaml`).

Layer 2 never generates factual claims — it retrieves a `CounterMove` by
technique id and phrases it around a verbatim span. See docs/DESIGN_RULES.md.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from app.taxonomy import Technique

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "counter_moves.yaml"


class CounterMove(BaseModel):
    id: Technique
    plain_name: str
    why_it_works: str
    reality_check: str
    source: str
    source_url: str
    """A followable link to `source`. Surfaced all the way to the UI so a
    reader can check the claim themselves — a citation nobody can follow is
    not much better than no citation."""
    counter_action: str

    @field_validator("plain_name", "why_it_works", "reality_check", "source", "source_url", "counter_action")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must not be blank")
        return v.strip()


@lru_cache(maxsize=1)
def load_counter_moves(path: Path = DATA_PATH) -> dict[Technique, CounterMove]:
    """Load and validate the counter-move table, keyed by technique.

    Raises if the file is missing an entry for any taxonomy technique, has
    an entry for a technique that doesn't exist, or any entry fails
    validation (most commonly: a blank `source`).
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a top-level YAML list")

    moves = [CounterMove.model_validate(entry) for entry in raw]

    seen_ids = [m.id for m in moves]
    if len(seen_ids) != len(set(seen_ids)):
        dupes = {t for t in seen_ids if seen_ids.count(t) > 1}
        raise ValueError(f"duplicate technique ids in {path}: {dupes}")

    by_id = {m.id: m for m in moves}

    all_techniques = set(Technique)
    covered = set(by_id.keys())
    missing = all_techniques - covered
    if missing:
        raise ValueError(f"{path} is missing counter-moves for: {sorted(t.value for t in missing)}")

    return by_id


def get_counter_move(technique: Technique) -> CounterMove:
    return load_counter_moves()[technique]
