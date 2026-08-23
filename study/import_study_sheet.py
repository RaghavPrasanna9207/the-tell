"""Import completed per-participant sheets from export_study_sheet.py back
into `data/responses.jsonl`, in exactly the schema run_study.py itself
writes — so analyze_study.py needs no changes regardless of which front
end (CLI or spreadsheet) collected the data.

Stimulus identity is recovered from ROW ORDER (the sheet never contains
stimulus_id or kind — see export_study_sheet.py's docstring on why), so
this assumes the 6 response rows are still in the order they were
exported in. Arm and participant id are recovered from the filename,
which is never shown to the participant.

A sheet is only imported once all 6 rows have all three answers filled
in and valid; otherwise it's left in place (not moved, not partially
imported) with a warning, so it can just be re-run once complete. An
already-imported participant id (already present in responses.jsonl) is
skipped with a warning rather than double-counted.

Run: python study/import_study_sheet.py
"""

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook

from run_study import RESPONSE_CHOICES, RESPONSE_LABELS
from stimuli import STIMULI

PENDING_DIR = Path(__file__).parent / "data" / "sheets" / "pending"
COMPLETED_DIR = Path(__file__).parent / "data" / "sheets" / "completed"
RESPONSES_PATH = Path(__file__).parent / "data" / "responses.jsonl"

FILENAME_RE = re.compile(r"^study_(?P<pid>[0-9a-f]+)__(?P<arm>control|treatment)\.xlsx$")
LABEL_TO_CODE = dict(zip(RESPONSE_LABELS, RESPONSE_CHOICES))


def parse_sheet(path: Path) -> list[dict] | None:
    """Returns 6 validated response records, or None (with a printed
    reason) if the sheet isn't ready to import yet."""
    wb = load_workbook(path, data_only=True)
    if "Responses" not in wb.sheetnames:
        print(f"  SKIP {path.name}: no 'Responses' sheet found")
        return None

    ws = wb["Responses"]
    rows = list(ws.iter_rows(min_row=2, max_row=1 + len(STIMULI), values_only=True))
    if len(rows) != len(STIMULI):
        print(f"  SKIP {path.name}: expected {len(STIMULI)} response rows, found {len(rows)}")
        return None

    records = []
    for i, row in enumerate(rows):
        stimulus = STIMULI[i]
        response_label, confidence, warn_label = row[3], row[4], row[5]

        if response_label not in LABEL_TO_CODE:
            print(f"  SKIP {path.name}: row {i + 1} 'your_response' is missing or invalid ({response_label!r})")
            return None
        if confidence not in (1, 2, 3, 4, 5):
            print(f"  SKIP {path.name}: row {i + 1} 'confidence_1_to_5' is missing or invalid ({confidence!r})")
            return None
        if warn_label not in ("Yes", "No"):
            print(f"  SKIP {path.name}: row {i + 1} 'would_warn_others' is missing or invalid ({warn_label!r})")
            return None

        records.append(
            {
                "stimulus_id": stimulus["id"],
                "kind": stimulus["kind"],
                "response": LABEL_TO_CODE[response_label],
                "confidence": int(confidence),
                "warn_others": warn_label == "Yes",
            }
        )
    return records


def append_responses(participant_id: str, arm: str, records: list[dict]) -> None:
    RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with RESPONSES_PATH.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "participant_id": participant_id,
                        "arm": arm,
                        "stimulus_id": r["stimulus_id"],
                        "kind": r["kind"],
                        "response": r["response"],
                        "confidence": r["confidence"],
                        "warn_others": r["warn_others"],
                        "timestamp": now,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    if not PENDING_DIR.exists() or not list(PENDING_DIR.glob("*.xlsx")):
        print(f"No sheets found in {PENDING_DIR}. Run export_study_sheet.py first.")
        sys.exit(0)

    history_ids = set()
    if RESPONSES_PATH.exists():
        with RESPONSES_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history_ids.add(json.loads(line)["participant_id"])

    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    imported = 0
    for path in sorted(PENDING_DIR.glob("*.xlsx")):
        m = FILENAME_RE.match(path.name)
        if not m:
            print(f"  SKIP {path.name}: filename doesn't match the expected study_<id>__<arm>.xlsx pattern")
            continue
        participant_id, arm = m.group("pid"), m.group("arm")

        if participant_id in history_ids:
            print(f"  SKIP {path.name}: participant {participant_id} already imported")
            continue

        records = parse_sheet(path)
        if records is None:
            continue

        append_responses(participant_id, arm, records)
        shutil.move(str(path), str(COMPLETED_DIR / path.name))
        print(f"  OK {path.name}  (participant {participant_id}, arm {arm})")
        imported += 1

    print(f"\nImported {imported} participant(s) into {RESPONSES_PATH}")


if __name__ == "__main__":
    main()
