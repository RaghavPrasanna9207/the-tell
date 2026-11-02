"""Publish the distilled student to the Hugging Face Hub.

`backend/models/` is gitignored (~574MB of checkpoint), so a clone has no
weights and the deployed image has nothing to load. Publishing them is what
makes the cascade gate work anywhere other than the machine that trained it.

The model card is written from this project's own eval output rather than a
template, including the failure modes — a model card that lists only the
headline metric is the same mistake as a README that reports only macro-F1.

Requires: huggingface-cli login (or HF_TOKEN in the environment).

Run: python eval/publish_student.py --repo <user>/the-tell-student-modernbert
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.student import STUDENT_MODEL_DIR
from app.taxonomy import Technique

MODEL_CARD = """---
license: mit
library_name: transformers
pipeline_tag: text-classification
tags:
  - multi-label-classification
  - scam-detection
  - modernbert
  - distillation
---

# The Tell — distilled manipulation-technique classifier

A 150M-parameter ModernBERT-base fine-tune that tags a message with the
manipulation techniques it uses. Distilled from a Qwen2.5-7B teacher, then
deployed as a **high-recall gate** in front of that teacher rather than as a
replacement for it: it decides whether the expensive model is worth waking,
and roughly 71% of messages resolve without it.

It classifies *techniques*, not "scam / not scam". The product it belongs to
explains the manipulation being attempted instead of just flagging a message.

## Labels ({n} techniques)

{labels}

## Intended use, and what it cannot do

Built as a pre-filter for a scam-explanation tool targeting Indian
digital-arrest and UPI scam scripts. It has **no notion of verbatim spans**,
so it can never produce a grounded explanation on its own — that only ever
comes from the teacher. Downstream text that quotes a message must not be
built from this model's output.

**Never render its silence as "safe."** Measured end to end, the gate silences
9 of 69 scam messages in the held-out set. Absence of a detection is absence of
evidence.

## Results (held-out threshold calibration, N=250 gold)

Per-technique thresholds are chosen by 5-fold CV — each fold's threshold is
picked using only the other folds, so every message is scored by a threshold
that never saw its own label. In-sample numbers run noticeably higher and are
not reported here.

| System | Macro-F1 | Latency | Size |
|---|---|---|---|
| TF-IDF + LR baseline | 0.41 | negligible | negligible |
| Teacher (Qwen2.5-7B) | see repo | slow | ~4.5GB |
| **This model** | **~0.48** | ~100-300ms | 574MB / 150M params |

Read as a latency/size trade, not a free accuracy win. On the four
highest-support techniques the student beats the teacher; the teacher's edge is
concentrated in techniques with single-digit training support.

## Known failure modes

These are measured, not hypothetical:

- **`payment_irreversibility` collapses to F1 0.00 held-out.** The model learned
  "rupee amount + payment verb" as a surface cue, which is shared by real scams,
  legitimate bank notices, *and* anti-scam PSAs. It false-positives on a
  legitimate OTP notice, an electricity bill reminder, and a bank's own warning
  telling users never to share their PIN.
- **`channel_switch` barely learned** (F1 0.12) — only 13 positive training
  examples.
- Low-support techniques are unstable run to run. A single misprediction on a
  3-support class swings its F1 by 0.33 and drags the macro average with it.
  Read the per-technique rows, not the macro line.

## Training data, and its limits

Teacher-labeled (silver) messages from a public SMS spam corpus, plus 78
hand-written India-specific messages authored to cover techniques the corpus
did not contain. The gold evaluation set is excluded by id from training — an
earlier run leaked 190 of 250 gold messages into training and was retrained
after that was found.

The evaluation set is 250 messages, 190 of them 2000s-era UK SMS spam, with 45
of 69 positive examples written and labeled by the same author who designed the
taxonomy. Inter-annotator agreement on an 80-message subset is Cohen's
kappa 0.68. **These numbers describe performance against that distribution, not
against live Indian scam traffic.**

## Source

Full pipeline, evals, error analysis and the human study:
https://github.com/RaghavPrasanna9207/the-tell
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. yourname/the-tell-student-modernbert")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the card locally, upload nothing")
    args = ap.parse_args()

    if not STUDENT_MODEL_DIR.exists():
        print(f"No student model at {STUDENT_MODEL_DIR} — run eval/distill.py first.")
        sys.exit(1)

    labels = "\n".join(f"- `{t.value}`" for t in Technique)
    card = MODEL_CARD.format(n=len(list(Technique)), labels=labels)
    card_path = STUDENT_MODEL_DIR / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"Wrote model card to {card_path}")

    if args.dry_run:
        print("--dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    # The training checkpoints subdirectory is intermediate Trainer state, not
    # part of the model — several hundred MB of optimizer shards nobody needs.
    api.upload_folder(
        folder_path=str(STUDENT_MODEL_DIR),
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=["checkpoints/*", "**/optimizer.pt", "**/scheduler.pt", "**/rng_state.pth"],
    )
    print(f"Published: https://huggingface.co/{args.repo}")
    print(f"\nNow set STUDENT_HF_REPO={args.repo} for the deployed image.")


if __name__ == "__main__":
    main()
