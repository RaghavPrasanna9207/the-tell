"""Export per-participant response sheets — a spreadsheet front end for the
same pre-registered study `run_study.py` runs over the CLI (see
PREREGISTRATION.md). Same stimuli, same fixed order, same arm-assignment
logic, same response schema; only the medium differs. Nothing about the
pre-registered design is medium-specific, so this needs no amendment to
PREREGISTRATION.md.

Each participant sees only what their assigned arm would show — the flag
(control) or the full named-technique explanation (treatment) — for each of
the 6 fixed stimuli, in fixed order. Deliberately NOT written anywhere
visible: `stimulus_id` or `kind` (e.g. "S1_digital_arrest" would itself give
away that message 1 is a scam before the participant reads it) and no
visible "control"/"treatment" label (the CLI never announces the arm to the
participant either). Row order alone is how `import_study_sheet.py` maps
answers back to stimuli, since the order is fixed and known.

Answers are entered as single letters/digits (C/R/I/V/S for response, Y/N
for would_warn_others, 1-5 for confidence) via a dropdown, not free text —
typed multi-word labels are what got silently rejected by Excel's list
validation and reverted to blank during the pilot.

Two output modes:
  --count N (default)        one .xlsx file per participant
  --count N --single-file    one .xlsx with N tabs (P01..PNN), one per
                              participant, plus a hidden "_index" tab (not
                              shown in the tab bar) mapping each tab to its
                              participant id and arm — read only by
                              import_study_sheet.py.

Arm assignment reuses `next_arm`/`STUDY_SEED` from run_study.py, indexed by
(completed participants in responses.jsonl) + (participants already
exported but not yet imported, counted from pending/) — so exporting a
batch ahead of any being returned still keeps blocks balanced, without
needing a separate reservation ledger.

Requires stimuli_cache.json — run build_stimuli_cache.py first.

Run: python study/export_study_sheet.py [--count N] [--single-file]
"""

import argparse
import re
import sys
import uuid
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from run_study import RESPONSE_LETTER_CODES, load_arm_history, load_cache, next_arm
from stimuli import STIMULI

PENDING_DIR = Path(__file__).parent / "data" / "sheets" / "pending"
RESPONSES_PATH = Path(__file__).parent / "data" / "responses.jsonl"

FILENAME_RE = re.compile(r"^study_(?P<pid>[0-9a-f]+)__(?P<arm>control|treatment)\.xlsx$")

HEADER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
HEADER_FONT = Font(bold=True)

DATA_START_ROW = 3  # row 1 = legend banner, row 2 = column headers
DATA_END_ROW = DATA_START_ROW - 1 + len(STIMULI)

RESPONSE_LEGEND = [
    ("C", "Comply — do what the message asks"),
    ("R", "Reply to ask a question"),
    ("I", "Ignore"),
    ("V", "Verify independently (contact them another way you already trust)"),
    ("S", "Call someone you trust"),
]
RESPONSE_LETTERS = ",".join(code for code, _ in RESPONSE_LEGEND)
RESPONSE_BANNER = (
    "Response letters: " + "  ".join(f"{k}={v.split(' — ')[0].split(' (')[0]}" for k, v in RESPONSE_LEGEND)
    + "   |   Warn: Y=Yes  N=No"
)

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


def _write_instructions(ws, *, batch: bool, n: int = 1) -> None:
    ws["A1"] = "Before you start"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = CONSENT_TEXT
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")

    ws["A4"] = "How to fill this in"
    ws["A4"].font = Font(bold=True)
    if batch:
        how_to = (
            f"This workbook has {n} tabs (P01..P{n:02d}) at the bottom, one per participant. "
            "Each participant fills in exactly one tab — go to the tab you've been assigned, "
            "read each message and whatever is shown below it, then pick one answer in each of "
            "the three response columns using the dropdown."
        )
    else:
        how_to = (
            "Go to the 'Responses' sheet. For each of the 6 messages, read the message and "
            "whatever is shown below it, then pick one answer in each of the three response "
            "columns using the dropdown."
        )
    ws["A5"] = how_to
    ws["A5"].alignment = Alignment(wrap_text=True, vertical="top")

    ws["A7"] = "Response letters"
    ws["A7"].font = Font(bold=True)
    row = 8
    for letter, meaning in RESPONSE_LEGEND:
        ws.cell(row=row, column=1, value=letter).font = Font(bold=True)
        ws.cell(row=row, column=2, value=meaning)
        row += 1
    row += 1
    ws.cell(row=row, column=1, value="Y").font = Font(bold=True)
    ws.cell(row=row, column=2, value="Yes, I would warn someone else about this message")
    row += 1
    ws.cell(row=row, column=1, value="N").font = Font(bold=True)
    ws.cell(row=row, column=2, value="No, I would not")

    ws.column_dimensions["A"].width = 90
    ws.column_dimensions["B"].width = 60
    for r in (2, 5):
        ws.row_dimensions[r].height = 60


def _write_responses_sheet(ws, arm: str, cache: dict) -> None:
    ws.append([RESPONSE_BANNER])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=6)
    banner_cell = ws.cell(row=1, column=1)
    banner_cell.font = Font(italic=True)
    banner_cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = 45

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
        cell = ws.cell(row=2, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    for i, stimulus in enumerate(STIMULI, start=1):
        entry = cache[stimulus["id"]]
        system_shows = _render_control(entry) if arm == "control" else _render_treatment(entry)
        ws.append([i, entry["text"], system_shows, "", "", ""])

    for row in ws.iter_rows(min_row=DATA_START_ROW, max_row=DATA_END_ROW):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        row[2].alignment = Alignment(wrap_text=True, vertical="top")

    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 55
    ws.column_dimensions["C"].width = 55
    ws.column_dimensions["D"].width = 16
    ws.column_dimensions["E"].width = 16
    ws.column_dimensions["F"].width = 16
    ws.freeze_panes = f"A{DATA_START_ROW}"

    response_dv = DataValidation(type="list", formula1=f'"{RESPONSE_LETTERS}"', allow_blank=True)
    response_dv.add(f"D{DATA_START_ROW}:D{DATA_END_ROW}")
    ws.add_data_validation(response_dv)

    confidence_dv = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    confidence_dv.add(f"E{DATA_START_ROW}:E{DATA_END_ROW}")
    ws.add_data_validation(confidence_dv)

    warn_dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    warn_dv.add(f"F{DATA_START_ROW}:F{DATA_END_ROW}")
    ws.add_data_validation(warn_dv)


def build_sheet(participant_id: str, arm: str, cache: dict) -> Workbook:
    wb = Workbook()
    instr = wb.active
    instr.title = "Instructions"
    _write_instructions(instr, batch=False)
    ws = wb.create_sheet("Responses")
    _write_responses_sheet(ws, arm, cache)
    return wb


def build_batch_workbook(assignments: list[tuple[str, str]], cache: dict) -> Workbook:
    """assignments: list of (participant_id, arm), one per tab, in order."""
    wb = Workbook()
    instr = wb.active
    instr.title = "Instructions"
    _write_instructions(instr, batch=True, n=len(assignments))

    index_rows = []
    for i, (participant_id, arm) in enumerate(assignments, start=1):
        tab_name = f"P{i:02d}"
        ws = wb.create_sheet(tab_name)
        _write_responses_sheet(ws, arm, cache)
        index_rows.append((tab_name, participant_id, arm))

    idx_ws = wb.create_sheet("_index")
    idx_ws.append(["tab_name", "participant_id", "arm"])
    for row in index_rows:
        idx_ws.append(list(row))
    idx_ws.sheet_state = "hidden"

    return wb


def _count_pending_participants() -> int:
    count = 0
    for path in PENDING_DIR.glob("*.xlsx"):
        if path.name.startswith("~$"):
            continue
        if FILENAME_RE.match(path.name):
            count += 1
            continue
        try:
            wb = load_workbook(path, read_only=True)
        except Exception:
            continue
        if "_index" in wb.sheetnames:
            count += wb["_index"].max_row - 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1, help="How many participants to export")
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Bundle all exported participants into one workbook (one tab each) instead of one file per participant",
    )
    args = parser.parse_args()

    cache_path = Path(__file__).parent / "stimuli_cache.json"
    if not cache_path.exists():
        print(f"ERROR: {cache_path} not found. Run `python study/build_stimuli_cache.py` first.")
        sys.exit(1)
    cache = load_cache()

    history = load_arm_history(RESPONSES_PATH)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    already_pending = _count_pending_participants()
    start_index = len(history) + already_pending

    print(f"{len(history)} completed, {already_pending} already pending. Exporting {args.count} more.\n")

    assignments = []
    for i in range(args.count):
        arm = next_arm(start_index + i)
        assignments.append((uuid.uuid4().hex[:12], arm))

    if args.single_file:
        wb = build_batch_workbook(assignments, cache)
        out_path = PENDING_DIR / f"study_batch_{uuid.uuid4().hex[:8]}.xlsx"
        wb.save(out_path)
        print(f"  {out_path.name}  ({len(assignments)} participants, tabs P01..P{len(assignments):02d})\n")
        for i, (pid, arm) in enumerate(assignments, start=1):
            print(f"    P{i:02d} -> participant {pid}, arm {arm}")
        print(
            "\nGive each participant only their own tab (e.g. screen-share just that tab, or copy "
            "it out to its own file before sending) — the tab bar lists every other participant's "
            "tab too, but not their arm or id. Once all tabs are filled in, leave the workbook in "
            "place and run:\n  python study/import_study_sheet.py"
        )
    else:
        for participant_id, arm in assignments:
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
