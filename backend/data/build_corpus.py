"""Normalize raw corpus sources into one canonical JSONL schema and dedupe.

Sources:
  - corpus/raw/SMSSpamCollection   (UCI SMS Spam Collection, tab-separated
    label\\ttext; ham/spam only, generic English marketing spam — bulk
    volume and benign/negative examples, not India-specific)
  - corpus/raw/handcrafted_india.jsonl  (hand-authored, grounded in the
    documented digital-arrest / UPI scam script — this is where the
    taxonomy actually gets signal; see docs/DESIGN_RULES.md)

Output schema (one JSON object per line):
    id: str                    stable id, prefixed by source
    text: str                  the message
    source: str                which raw file this came from
    ham_spam: str | null       "ham"/"spam" if the source provides it (UCI only)
    draft_labels: list[str] | null   author-intent technique ids, NOT gold —
                                     see docs/DESIGN_RULES.md and the plan's hand-labeling
                                     step for the actual annotation pass

Run: python data/build_corpus.py
"""

import json
from pathlib import Path

RAW_DIR = Path(__file__).parent / "corpus" / "raw"
PROCESSED_DIR = Path(__file__).parent / "corpus" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "corpus.jsonl"


def load_uci_sms_spam() -> list[dict]:
    path = RAW_DIR / "SMSSpamCollection"
    if not path.exists():
        print(f"  (skipping - {path} not found)")
        return []

    records = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            label, _, text = line.partition("\t")
            records.append(
                {
                    "id": f"uci-{i:05d}",
                    "text": text,
                    "source": "uci_sms_spam",
                    "ham_spam": label,
                    "draft_labels": None,
                }
            )
    return records


def load_handcrafted_india() -> list[dict]:
    path = RAW_DIR / "handcrafted_india.jsonl"
    if not path.exists():
        print(f"  (skipping - {path} not found)")
        return []

    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            records.append(
                {
                    "id": entry["id"],
                    "text": entry["text"],
                    "source": entry.get("source", "handcrafted_india"),
                    "ham_spam": None,
                    "draft_labels": entry.get("draft_labels", []),
                }
            )
    return records


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped = []
    for r in records:
        key = r["text"].strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def main() -> None:
    print("Loading UCI SMS Spam Collection...")
    uci = load_uci_sms_spam()
    print(f"  {len(uci)} messages")

    print("Loading hand-crafted India corpus...")
    india = load_handcrafted_india()
    print(f"  {len(india)} messages")

    all_records = uci + india
    before = len(all_records)
    deduped = dedupe(all_records)
    after = len(deduped)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"Total before dedupe: {before}")
    print(f"Total after dedupe:  {after}  ({before - after} duplicates removed)")
    print(f"Wrote {OUTPUT_PATH}")

    ham = sum(1 for r in deduped if r["ham_spam"] == "ham")
    spam = sum(1 for r in deduped if r["ham_spam"] == "spam")
    handcrafted_scam = sum(1 for r in deduped if r["source"] == "handcrafted_india" and r["draft_labels"])
    handcrafted_clean = sum(
        1 for r in deduped if r["source"] == "handcrafted_india" and not r["draft_labels"]
    )
    print()
    print("Class distribution:")
    print(f"  UCI ham:               {ham}")
    print(f"  UCI spam:              {spam}")
    print(f"  Handcrafted (scam):    {handcrafted_scam}")
    print(f"  Handcrafted (benign):  {handcrafted_clean}")


if __name__ == "__main__":
    main()
