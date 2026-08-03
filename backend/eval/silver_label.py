"""Label the corpus with the local teacher model to produce silver labels
for distillation (see plan Week 2, step 2: ~1000 silver labels).

Uses the same sampling function and seed as `eval/label.py`'s gold set
(target=250, seed=42). Because `build_sample` shuffles the non-handcrafted
pool once and then slices, a larger target is always a superset of a
smaller one for the same seed — so the ~250-message gold set ends up a
strict subset of this ~1000-message silver sample. That overlap is what
lets teacher-vs-gold agreement be measured later.

Resumable: skips ids already present in the output file, so a Ctrl+C or a
crashed Ollama server mid-run doesn't lose progress. Run it again and it
picks up where it left off.

Run: python eval/silver_label.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classify import classify
from app.llm import OllamaUnavailableError, check_available
from eval.label import build_sample, load_corpus

TARGET = 1000
SEED = 42
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "corpus" / "processed" / "silver_labels.jsonl"


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    check_available()

    records = load_corpus()
    sample = build_sample(records, target=TARGET, seed=SEED)

    existing = load_existing_ids(OUTPUT_PATH)
    remaining = [r for r in sample if r["id"] not in existing]

    print(f"Sample size: {len(sample)}  |  Already labeled: {len(existing)}  |  Remaining: {len(remaining)}")
    print(f"Saving to: {OUTPUT_PATH}")

    if not remaining:
        print("Nothing left to label in this sample.")
        return

    total_elapsed = 0.0
    n_dropped_spans = 0
    n_errors = 0
    start = time.monotonic()

    for i, record in enumerate(remaining, start=1):
        try:
            result = classify(record["text"])
        except OllamaUnavailableError:
            raise
        except Exception as exc:
            n_errors += 1
            print(f"  [{i}/{len(remaining)}] ERROR on {record['id']}: {exc}")
            continue

        labels = [d.technique.value for d in result.analysis.detections]
        n_dropped_spans += len(result.dropped_spans)
        total_elapsed += result.generation.elapsed_seconds

        append_record(
            OUTPUT_PATH,
            {
                "id": record["id"],
                "text": record["text"],
                "source": record["source"],
                "labels": labels,
            },
        )

        if i % 20 == 0 or i == len(remaining):
            avg = total_elapsed / i
            eta_min = avg * (len(remaining) - i) / 60
            print(
                f"  [{i}/{len(remaining)}] avg {avg:.1f}s/msg, ETA {eta_min:.1f} min, "
                f"dropped_spans={n_dropped_spans}, errors={n_errors}"
            )

    wall_clock_min = (time.monotonic() - start) / 60
    n_labeled = len(existing) + len(remaining) - n_errors
    print(f"\nDone. {n_labeled} messages labeled in {OUTPUT_PATH}")
    print(f"Wall-clock this run: {wall_clock_min:.1f} min | dropped_spans={n_dropped_spans} | errors={n_errors}")


if __name__ == "__main__":
    main()
