"""Interactive CLI data-collection harness for the pre-registered study
(see PREREGISTRATION.md). One run = one participant session.

Presents the six fixed stimuli, same order, to every participant. The arm
(control = binary flag only, treatment = full named-technique explanation)
is assigned via permuted block randomization (block size 2: each block
contains exactly one control and one treatment, in random order) so arms
stay balanced at the small N this study targets (~10 per arm) while each
participant's assignment is still randomly drawn, per the pre-registration.
The block shuffle is seeded from STUDY_SEED + block index, so assignment
is reproducible across runs without needing to persist RNG state between
process invocations — only the count of participants assigned so far
(read back from the response log) determines the next arm.

No raw message text or PII is written to the response log — only
participant id (random, unlinked), arm, stimulus id, response code,
confidence, and timestamp. See PREREGISTRATION.md "Data handling".

Requires stimuli_cache.json — run build_stimuli_cache.py first.

Run: python study/run_study.py
"""

import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from stimuli import STIMULI

STUDY_SEED = 20260803  # fixed — see module docstring on block randomization

CACHE_PATH = Path(__file__).parent / "stimuli_cache.json"
RESPONSES_PATH = Path(__file__).parent / "data" / "responses.jsonl"

RESPONSE_CHOICES = ["comply", "reply", "ignore", "verify_independently", "call_someone"]
RESPONSE_LABELS = [
    "Comply",
    "Reply to ask a question",
    "Ignore",
    "Verify independently",
    "Call someone",
]

# Single-letter codes for the spreadsheet front end (export_study_sheet.py /
# import_study_sheet.py) — free-typed multi-word labels are what silently got
# rejected by Excel's dropdown validation and reverted to blank during the
# pilot, so the sheet asks for one letter per answer instead.
RESPONSE_LETTER_CODES = {
    "C": "comply",
    "R": "reply",
    "I": "ignore",
    "V": "verify_independently",
    "S": "call_someone",
}


def load_cache() -> dict:
    if not CACHE_PATH.exists():
        print(f"ERROR: {CACHE_PATH} not found. Run `python study/build_stimuli_cache.py` first.")
        sys.exit(1)
    entries = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {e["id"]: e for e in entries}


def load_arm_history(path: Path) -> list[str]:
    """Arm assigned to each participant so far, in first-seen order."""
    if not path.exists():
        return []
    seen: dict[str, str] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r["participant_id"]
            if pid not in seen:
                seen[pid] = r["arm"]
                order.append(pid)
    return [seen[pid] for pid in order]


def next_arm(n_assigned_so_far: int) -> str:
    block_index, position_in_block = divmod(n_assigned_so_far, 2)
    block = ["control", "treatment"]
    random.Random(STUDY_SEED + block_index).shuffle(block)
    return block[position_in_block]


def parse_choice(raw: str) -> str:
    """Raises ValueError on invalid input."""
    i = int(raw.strip())
    if not (1 <= i <= len(RESPONSE_CHOICES)):
        raise ValueError(f"{i} is not a valid choice (1-{len(RESPONSE_CHOICES)})")
    return RESPONSE_CHOICES[i - 1]


def parse_confidence(raw: str) -> int:
    i = int(raw.strip())
    if not (1 <= i <= 5):
        raise ValueError(f"{i} is not a valid confidence (1-5)")
    return i


def parse_yes_no(raw: str) -> bool:
    v = raw.strip().lower()
    if v in ("y", "yes"):
        return True
    if v in ("n", "no"):
        return False
    raise ValueError("enter y or n")


def prompt_until_valid(prompt: str, parser):
    while True:
        raw = input(prompt)
        try:
            return parser(raw)
        except ValueError as e:
            print(f"  Invalid input: {e}")


def render_control(entry: dict) -> None:
    if entry["is_clean"]:
        print("\n[No warning was flagged for this message.]")
    else:
        print("\n[WARNING: This looks like a scam.]")


def render_treatment(entry: dict) -> None:
    if entry["is_clean"]:
        print("\nNo manipulation techniques detected.")
        return
    for card in entry["cards"]:
        print(f"\n>> {card['plain_name']} — \"{card['span']}\"")
        print(f"   {card['explanation']}")
        print(f"   Source: {card['source']}")


def append_response(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    cache = load_cache()
    history = load_arm_history(RESPONSES_PATH)
    arm = next_arm(len(history))
    participant_id = uuid.uuid4().hex[:12]

    print(f"Participant: {participant_id}  |  Arm: {arm}")
    print("You'll see 6 messages, one at a time. For each, choose what you would do.\n")

    for i, stimulus in enumerate(STIMULI, start=1):
        entry = cache[stimulus["id"]]
        print("\n" + "=" * 70)
        print(f"[{i}/6]\n")
        print(entry["text"])

        if arm == "control":
            render_control(entry)
        else:
            render_treatment(entry)

        print("\nWhat would you do?")
        for j, label in enumerate(RESPONSE_LABELS, start=1):
            print(f"  {j}. {label}")
        choice = prompt_until_valid("> ", parse_choice)
        confidence = prompt_until_valid("How confident are you in that judgment? (1-5) > ", parse_confidence)
        warn_others = prompt_until_valid("Would you warn someone else about this message? (y/n) > ", parse_yes_no)
        # Corrected H2 measure (PREREGISTRATION.md amendment 2026-09-08). Asked
        # LAST, deliberately: the original three questions keep their exact
        # order and wording so the 120 already-collected responses stay
        # comparable, and asking "is this genuine?" earlier would prime the
        # action choice that H1 depends on.
        believes_genuine = prompt_until_valid("Do you think this message is genuine? (y/n) > ", parse_yes_no)

        append_response(
            RESPONSES_PATH,
            {
                "participant_id": participant_id,
                "arm": arm,
                "stimulus_id": stimulus["id"],
                "kind": stimulus["kind"],
                "response": choice,
                "confidence": confidence,
                "warn_others": warn_others,
                "believes_genuine": believes_genuine,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    print(f"\nDone. Thank you. Responses saved under participant id {participant_id}.")


if __name__ == "__main__":
    main()
