"""Import completed response sheets from export_study_sheet.py back into
`data/responses.jsonl`, in exactly the schema run_study.py itself writes —
so analyze_study.py needs no changes regardless of which front end (CLI or
spreadsheet) collected the data.

Handles both export modes:
  - one file per participant (study_<id>__<arm>.xlsx, arm/id from filename)
  - one workbook with many tabs (study_batch_*.xlsx, arm/id per tab read
    from that workbook's hidden "_index" sheet)

Stimulus identity is recovered from ROW ORDER (the sheet never contains
stimulus_id or kind — see export_study_sheet.py's docstring on why), so
this assumes the 6 response rows are still in the order they were exported
in. Response letters (C/R/I/V/S) and warn letters (Y/N) are decoded back to
the same codes run_study.py uses.

A sheet/tab is only imported once all 6 rows have all three answers filled
in and valid; otherwise it's left in place (not moved, not partially
imported) with a warning, so it can just be re-run once complete. A batch
workbook is moved to completed/ only once every tab in it has been
imported — tabs already imported in a prior run are skipped, not
re-imported, so a partially-completed batch can be re-run safely. An
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

from run_study import RESPONSE_LETTER_CODES
from stimuli import STIMULI

PENDING_DIR = Path(__file__).parent / "data" / "sheets" / "pending"
COMPLETED_DIR = Path(__file__).parent / "data" / "sheets" / "completed"
RESPONSES_PATH = Path(__file__).parent / "data" / "responses.jsonl"

FILENAME_RE = re.compile(r"^study_(?P<pid>[0-9a-f]+)__(?P<arm>control|treatment)\.xlsx$")
WARN_LETTER_TO_BOOL = {"Y": True, "N": False}

DATA_START_ROW = 3  # row 1 = legend banner, row 2 = column headers
DATA_END_ROW = DATA_START_ROW - 1 + len(STIMULI)


def _parse_responses_ws(ws, label: str) -> list[dict] | None:
    """Returns 6 validated response records, or None (with a printed
    reason) if the sheet isn't ready to import yet."""
    rows = list(ws.iter_rows(min_row=DATA_START_ROW, max_row=DATA_END_ROW, values_only=True))
    if len(rows) != len(STIMULI):
        print(f"  SKIP {label}: expected {len(STIMULI)} response rows, found {len(rows)}")
        return None

    records = []
    for i, row in enumerate(rows):
        stimulus = STIMULI[i]
        response_raw, confidence, warn_raw = row[3], row[4], row[5]

        response_letter = response_raw.strip().upper() if isinstance(response_raw, str) else response_raw
        if response_letter not in RESPONSE_LETTER_CODES:
            print(f"  SKIP {label}: row {i + 1} 'your_response' is missing or invalid ({response_raw!r})")
            return None

        if confidence not in (1, 2, 3, 4, 5):
            print(f"  SKIP {label}: row {i + 1} 'confidence_1_to_5' is missing or invalid ({confidence!r})")
            return None

        if isinstance(warn_raw, bool):
            warn_others = warn_raw
        else:
            warn_letter = warn_raw.strip().upper() if isinstance(warn_raw, str) else warn_raw
            if warn_letter not in WARN_LETTER_TO_BOOL:
                print(f"  SKIP {label}: row {i + 1} 'would_warn_others' is missing or invalid ({warn_raw!r})")
                return None
            warn_others = WARN_LETTER_TO_BOOL[warn_letter]

        records.append(
            {
                "stimulus_id": stimulus["id"],
                "kind": stimulus["kind"],
                "response": RESPONSE_LETTER_CODES[response_letter],
                "confidence": int(confidence),
                "warn_others": warn_others,
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


def _import_batch(path: Path, wb, history_ids: set) -> int:
    idx_ws = wb["_index"]
    entries = list(idx_ws.iter_rows(min_row=2, values_only=True))
    imported = 0
    remaining = 0
    for tab_name, participant_id, arm in entries:
        if participant_id in history_ids:
            continue
        if tab_name not in wb.sheetnames:
            print(f"  SKIP {path.name}:{tab_name}: tab not found")
            remaining += 1
            continue
        records = _parse_responses_ws(wb[tab_name], f"{path.name}:{tab_name}")
        if records is None:
            remaining += 1
            continue
        append_responses(participant_id, arm, records)
        history_ids.add(participant_id)
        print(f"  OK {path.name}:{tab_name}  (participant {participant_id}, arm {arm})")
        imported += 1

    if remaining == 0:
        shutil.move(str(path), str(COMPLETED_DIR / path.name))
    else:
        print(f"  {path.name}: {remaining} tab(s) still incomplete, left in pending")
    return imported


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
        if path.name.startswith("~$"):
            continue

        wb = load_workbook(path, data_only=True)

        if "_index" in wb.sheetnames:
            imported += _import_batch(path, wb, history_ids)
            continue

        m = FILENAME_RE.match(path.name)
        if not m:
            print(f"  SKIP {path.name}: filename doesn't match the expected study_<id>__<arm>.xlsx pattern")
            continue
        participant_id, arm = m.group("pid"), m.group("arm")

        if participant_id in history_ids:
            print(f"  SKIP {path.name}: participant {participant_id} already imported")
            continue
        if "Responses" not in wb.sheetnames:
            print(f"  SKIP {path.name}: no 'Responses' sheet found")
            continue

        records = _parse_responses_ws(wb["Responses"], path.name)
        if records is None:
            continue

        append_responses(participant_id, arm, records)
        history_ids.add(participant_id)
        shutil.move(str(path), str(COMPLETED_DIR / path.name))
        print(f"  OK {path.name}  (participant {participant_id}, arm {arm})")
        imported += 1

    print(f"\nImported {imported} participant(s) into {RESPONSES_PATH}")


if __name__ == "__main__":
    main()
