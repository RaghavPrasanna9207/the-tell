"""Interactive CLI for hand-labeling the gold evaluation set.

Presents one message at a time from a fixed, reproducible sample (all 60
hand-crafted India messages + a random sample of UCI messages to round out
to --target total), lets you tag it against the taxonomy, and saves
incrementally so nothing is lost if you stop partway through.

Two independent people labeling the SAME subset (same --seed) is how the
plan's Cohen's kappa gets computed — run this once as yourself, and have a
second person run it with --annotator second on the same machine or a copy
of the repo.

Usage:
    python eval/label.py                        # label as default annotator
    python eval/label.py --annotator second      # second annotator, same subset
    python eval/label.py --target 250            # change gold-set size
    python eval/label.py --kappa-subset 80        # only the first N messages
                                                   # (the subset both annotators
                                                   # actually overlap on)
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.taxonomy import Technique

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus" / "processed" / "corpus.jsonl"
LABELS_DIR = Path(__file__).parent.parent / "data" / "corpus" / "gold"

TECHNIQUES = list(Technique)


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        print(f"ERROR: {CORPUS_PATH} not found. Run `python data/build_corpus.py` first.")
        sys.exit(1)
    records = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_sample(records: list[dict], target: int, seed: int) -> list[dict]:
    """All handcrafted_india messages (the ones that matter for taxonomy
    coverage) plus a random, seeded sample of the rest to round out to
    `target`. Same seed => same sample => two annotators label the exact
    same messages, which is required for the kappa comparison to mean
    anything.
    """
    handcrafted = [r for r in records if r["source"] == "handcrafted_india"]
    rest = [r for r in records if r["source"] != "handcrafted_india"]

    rng = random.Random(seed)
    rng.shuffle(rest)

    n_from_rest = max(0, target - len(handcrafted))
    sample = handcrafted + rest[:n_from_rest]

    rng2 = random.Random(seed)
    rng2.shuffle(sample)
    return sample


def load_existing_labels(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    labels = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                labels[entry["id"]] = entry["labels"]
    return labels


def append_label(path: Path, record_id: str, text: str, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": record_id, "text": text, "labels": labels}, ensure_ascii=False) + "\n")


def print_menu() -> None:
    print()
    for i, t in enumerate(TECHNIQUES, start=1):
        print(f"  {i:>2}. {t.value}")
    print("   0. (none — no manipulation technique present)")
    print()
    print("Enter comma-separated numbers (e.g. '2,5'), '0' for none,")
    print("'s' to skip, 'q' to save and quit.")


def parse_input(raw: str) -> list[str] | None:
    """Returns None for skip, [] for 'none', or a list of technique values.
    Raises ValueError on bad input so the caller can re-prompt."""
    raw = raw.strip().lower()
    if raw == "s":
        return None
    if raw == "0":
        return []
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    labels = []
    for i in indices:
        if not (1 <= i <= len(TECHNIQUES)):
            raise ValueError(f"{i} is not a valid technique number (1-{len(TECHNIQUES)})")
        labels.append(TECHNIQUES[i - 1].value)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotator", default="primary", help="Label suffix, e.g. 'primary' or 'second'")
    parser.add_argument("--target", type=int, default=250, help="Total gold-set size (default 250)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed — keep fixed across annotators")
    parser.add_argument(
        "--kappa-subset",
        type=int,
        default=None,
        help="Only label the first N messages of the sample (for the inter-annotator agreement subset)",
    )
    args = parser.parse_args()

    output_path = LABELS_DIR / f"labels_{args.annotator}.jsonl"

    records = load_corpus()
    sample = build_sample(records, args.target, args.seed)
    if args.kappa_subset:
        sample = sample[: args.kappa_subset]

    existing = load_existing_labels(output_path)
    remaining = [r for r in sample if r["id"] not in existing]

    print(f"Annotator: {args.annotator}")
    print(f"Sample size: {len(sample)}  |  Already labeled: {len(existing)}  |  Remaining: {len(remaining)}")
    print(f"Saving to: {output_path}")

    if not remaining:
        print("Nothing left to label in this sample.")
        return

    for i, record in enumerate(remaining, start=1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(remaining)}]")
        print(f"\n{record['text']}\n")
        print_menu()

        while True:
            raw = input("> ")
            if raw.strip().lower() == "q":
                print(f"\nSaved. {len(existing)} labeled so far in {output_path}")
                return
            try:
                labels = parse_input(raw)
            except ValueError as e:
                print(f"  Invalid input: {e}")
                continue
            break

        if labels is None:  # skip
            continue

        append_label(output_path, record["id"], record["text"], labels)
        existing[record["id"]] = labels

    print(f"\nDone. {len(existing)} messages labeled in {output_path}")


if __name__ == "__main__":
    main()
