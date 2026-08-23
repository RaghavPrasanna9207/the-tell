"""Export one (or more) per-participant .xlsx response sheets — a
spreadsheet front end for the same pre-registered study `run_study.py`
runs over the CLI (see PREREGISTRATION.md). Same stimuli, same fixed
order, same arm-assignment logic, same response schema; only the medium
differs. Nothing about the pre-registered design is medium-specific, so
this needs no amendment to PREREGISTRATION.md.

Each sheet shows only what that one participant's assigned arm would see
— the flag (control) or the full named-technique explanation (treatment)
— for each of the 6 fixed stimuli, in fixed order. Deliberately NOT
written into the sheet: `stimulus_id` or `kind` (e.g. "S1_digital_arrest"
would itself give away that message 1 is a scam before the participant
reads it) and no visible "control"/"treatment" label (the CLI never
announces the arm to the participant either). Row order alone is how
`import_study_sheet.py` maps answers back to stimuli, since the order is
fixed and known.

Arm assignment reuses `next_arm`/`STUDY_SEED` from run_study.py, indexed
by (completed participants in responses.jsonl) + (sheets already
exported but not yet imported, counted from the pending/ directory) —
so exporting a batch ahead of any being returned still keeps blocks
balanced, without needing a separate reservation ledger.

Requires stimuli_cache.json — run build_stimuli_cache.py first.

Run: python study/export_study_sheet.py [--count N]
"""

import argparse
import sys
import uuid
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from run_study import RESPONSE_LABELS, load_arm_history, load_cache, next_arm
from stimuli import STIMULI

PENDING_DIR = Path(__file__).parent / "data" / "sheets" / "pending"
RESPONSES_PATH = Path(__file__).parent / "data" / "responses.jsonl"

HEADER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
HEADER_FONT = Font(bold=True)

CONSENT_TEXT = (
    "You're being asked to look at 6 short messages and say what you'd actually do "
    "with each one. Your answers are anonymous — nothing you write is linked to your "
    "name, and no personal information is collected. Answer with your honest, gut "
    "reaction; there's no 'correct' answer being scored. You can stop at any point by "
    "just not returning the sheet."
)


def _render_control(entry: dict) -> str:
    return "This looks like a scam." if not entry["is_clean"] else "No warning was flagged for this message."


def _render_treatment(entry: dict) -> str:
    if entry["is_clean"]:
        return "No manipulation techniques detected."
    parts = []
    for card in entry["cards"]:
        parts.append(f'{card["plain_name"]} — "{card["span"]}"\n{card["explanation"]}\nSource: {card["source"]}')
    return "\n\n".join(parts)


def build_sheet(participant_id: str, arm: str, cache: dict) -> Workbook:
    wb = Workbook()

    instr = wb.active
    instr.title = "Instructions"
    instr["A1"] = "Before you start"
    instr["A1"].font = Font(bold=True, size=13)
    instr["A2"] = CONSENT_TEXT
    instr["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    instr["A4"] = "How to fill this in"
    instr["A4"].font = Font(bold=True)
    instr["A5"] = "Go to the 'Responses' sheet. For each of the 6 messages, read the message and whatever is shown below it, then pick one answer in each of the three response columns using the dropdown."
    instr["A5"].alignment = Alignment(wrap_text=True, vertical="top")
    instr.column_dimensions["A"].width = 100
    for row in (2, 5):
        instr.row_dimensions[row].height = 60

    ws = wb.create_sheet("Responses")
    headers = [
        "message_number",
        "message",
        "system_shows",
        "your_response",
        "confidence_1_to_5",
        "would_warn_others",
    ]
    ws.append(headers)
    for col, _ in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    for i, stimulus in enumerate(STIMULI, start=1):
        entry = cache[stimulus["id"]]
        system_shows = _render_control(entry) if arm == "control" else _render_treatment(entry)
        ws.append([i, entry["text"], system_shows, "", "", ""])

    for row in ws.iter_rows(min_row=2, max_row=1 + len(STIMULI)):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        row[2].alignment = Alignment(wrap_text=True, vertical="top")

    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 55
    ws.column_dimensions["C"].width = 55
    ws.column_dimensions["D"].width = 24
    ws.column_dimensions["E"].width = 16
    ws.column_dimensions["F"].width = 16
    ws.freeze_panes = "A2"

    last_row = 1 + len(STIMULI)

    response_dv = DataValidation(type="list", formula1=f'"{",".join(RESPONSE_LABELS)}"', allow_blank=True)
    response_dv.add(f"D2:D{last_row}")
    ws.add_data_validation(response_dv)

    confidence_dv = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    confidence_dv.add(f"E2:E{last_row}")
    ws.add_data_validation(confidence_dv)

    warn_dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    warn_dv.add(f"F2:F{last_row}")
    ws.add_data_validation(warn_dv)

    return wb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1, help="How many participant sheets to export")
    args = parser.parse_args()

    cache_path = Path(__file__).parent / "stimuli_cache.json"
    if not cache_path.exists():
        print(f"ERROR: {cache_path} not found. Run `python study/build_stimuli_cache.py` first.")
        sys.exit(1)
    cache = load_cache()

    history = load_arm_history(RESPONSES_PATH)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    already_pending = len(list(PENDING_DIR.glob("*.xlsx")))
    start_index = len(history) + already_pending

    print(f"{len(history)} completed, {already_pending} already pending. Exporting {args.count} more.\n")
    for i in range(args.count):
        idx = start_index + i
        arm = next_arm(idx)
        participant_id = uuid.uuid4().hex[:12]
        wb = build_sheet(participant_id, arm, cache)
        out_path = PENDING_DIR / f"study_{participant_id}__{arm}.xlsx"
        wb.save(out_path)
        print(f"  {out_path.name}  (arm: {arm})")

    print(
        "\nHand each file to one participant (in person or via screen-share — the "
        "arm is only in the filename, never shown to them). Once filled in, leave "
        "the files in place and run:\n  python study/import_study_sheet.py"
    )


if __name__ == "__main__":
    main()
