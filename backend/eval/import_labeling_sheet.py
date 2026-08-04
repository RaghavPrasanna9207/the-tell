"""Convert a filled-in gold_labeling.xlsx back into labels_primary.jsonl.

Reads the "Labels" sheet: any non-empty cell in a technique column means
that technique applies to that row. Writes one JSON object per row to
labels_primary.jsonl, in the schema eval/label.py itself produces, so
nothing downstream (compare.py, distill.py, etc.) needs to change.

The previous labels_primary.jsonl is copied to labels_primary.jsonl.bak
before being overwritten.

Run: python eval/import_labeling_sheet.py
"""

import json
import shutil
import sys
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.taxonomy import Technique
from eval.label import LABELS_DIR

SHEET_PATH = LABELS_DIR / "gold_labeling.xlsx"
OUTPUT_PATH = LABELS_DIR / "labels_primary.jsonl"
BACKUP_PATH = LABELS_DIR / "labels_primary.jsonl.bak"

TECHNIQUE_VALUES = [t.value for t in Technique]


def main() -> None:
    if not SHEET_PATH.exists():
        print(f"ERROR: {SHEET_PATH} not found. Run eval/export_labeling_sheet.py first.")
        sys.exit(1)

    wb = load_workbook(SHEET_PATH, data_only=True)
    ws = wb["Labels"]

    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    technique_cols = {}
    for i, name in enumerate(header):
        if name in TECHNIQUE_VALUES:
            technique_cols[name] = i

    missing = set(TECHNIQUE_VALUES) - set(technique_cols)
    if missing:
        print(f"ERROR: sheet is missing technique columns: {missing}")
        sys.exit(1)
    if "none" not in header:
        print("ERROR: sheet is missing the 'none' column.")
        sys.exit(1)
    none_col = header.index("none")

    records = []
    skipped_unreviewed = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        record_id = row[0]
        if not record_id:
            continue
        text = row[2]
        filled = [name for name, i in technique_cols.items() if row[i]]
        none_marked = bool(row[none_col])

        if filled:
            if none_marked:
                print(f"WARNING: {record_id} has both 'none' and specific techniques marked — using the techniques.")
            labels = filled
        elif none_marked:
            labels = []
        else:
            skipped_unreviewed += 1
            continue

        records.append({"id": record_id, "text": text, "labels": labels})

    if OUTPUT_PATH.exists():
        shutil.copy(OUTPUT_PATH, BACKUP_PATH)
        print(f"Backed up previous labels to {BACKUP_PATH}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labeled = sum(1 for r in records if r["labels"])
    clean = sum(1 for r in records if not r["labels"])
    print(f"Wrote {len(records)} rows to {OUTPUT_PATH}")
    print(f"  with >=1 technique: {labeled}  |  none (clean): {clean}  |  not yet reviewed (skipped): {skipped_unreviewed}")


if __name__ == "__main__":
    main()
