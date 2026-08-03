"""Distill the Ollama teacher (Qwen2.5-7B) into a small deployable student
(ModernBERT-base) for multi-label technique classification.

HONESTY NOTE: this reads from `data/corpus/processed/silver_labels.jsonl`
(the teacher-labeled corpus) if it exists. That file doesn't exist yet —
it's produced by labeling the full corpus with the local Ollama teacher,
which hasn't run yet (see the plan's Week 2, step 2). Until then, this
script falls back to the ~60-message draft-labeled set purely as a
pipeline smoke test: it proves the training loop works end to end, not
that the resulting model is any good. Do not report numbers from a run
against the draft set as a real distillation result.

Run: python eval/distill.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np

# `datasets` (pyarrow) MUST be imported before `torch`/`transformers` on this
# machine — the reverse order reproducibly segfaults (torch's bundled CUDA/MKL
# DLLs conflict with pyarrow's when torch's native libs load first). Keep this
# import above the transformers import; don't let a linter "tidy" it away.
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from app.taxonomy import Technique
from eval.baseline import load_gold_as_arrays
from eval.gold import DRAFT_GOLD_WARNING, load_draft_gold
from eval.metrics import format_report, per_technique_metrics

MODEL_NAME = "answerdotai/ModernBERT-base"
SILVER_LABELS_PATH = Path(__file__).parent.parent / "data" / "corpus" / "processed" / "silver_labels.jsonl"
HANDCRAFTED_TRAIN_PATH = Path(__file__).parent.parent / "data" / "corpus" / "raw" / "handcrafted_india_train.jsonl"
STUDENT_MODEL_DIR = Path(__file__).parent.parent / "models" / "student"


def load_handcrafted_train() -> list[dict]:
    """Author-labeled messages written specifically to backstop distillation
    training (not silver-labeled, not used for eval). Separate from
    `handcrafted_india.jsonl` — that file is the gold-holdout set and must
    never be trained on; this one exists because the silver-labeled pool
    turned out to have near-zero positive coverage for most techniques (see
    the first real distillation run: only reciprocity_hook/fake_scarcity/
    manufactured_urgency had meaningful support, because the UCI SMS corpus
    barely contains India-specific manipulation language)."""
    if not HANDCRAFTED_TRAIN_PATH.exists():
        return []
    valid = {t.value for t in Technique}
    records = []
    with HANDCRAFTED_TRAIN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for label in entry["draft_labels"]:
                if label not in valid:
                    raise ValueError(f"{entry['id']} has invalid label {label!r}")
            records.append({"text": entry["text"], "labels": entry["draft_labels"]})
    return records


def load_training_data() -> tuple[list[str], np.ndarray, list[str], bool]:
    """Returns (texts, y_true, technique_names, is_smoke_test).

    IMPORTANT: silver records with source "handcrafted_india" are excluded
    from the trainable pool here. Those are the exact same messages
    `eval.gold.load_draft_gold()` returns as the gold-holdout eval set — if
    the student trained on them, `evaluate_on_true_gold` below would be
    measuring memorization, not generalization. They're held out entirely,
    never randomly split in, so the gold numbers this script reports are a
    real holdout.
    """
    if SILVER_LABELS_PATH.exists():
        records = []
        with SILVER_LABELS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        n_before = len(records)
        records = [r for r in records if r.get("source") != "handcrafted_india"]
        n_held_out = n_before - len(records)
        print(
            f"Held out {n_held_out} handcrafted_india silver records from training "
            f"(reserved for the true gold-holdout eval) — {len(records)} remain trainable."
        )

        texts = [r["text"] for r in records]
        labels = [r["labels"] for r in records]

        extra = load_handcrafted_train()
        if extra:
            print(f"Adding {len(extra)} hand-labeled records from {HANDCRAFTED_TRAIN_PATH.name} to training.")
            texts += [r["text"] for r in extra]
            labels += [r["labels"] for r in extra]

        technique_names = [t.value for t in Technique]
        from sklearn.preprocessing import MultiLabelBinarizer

        mlb = MultiLabelBinarizer(classes=technique_names)
        y_true = mlb.fit_transform(labels)
        return texts, y_true, technique_names, False

    print(f"WARNING: {SILVER_LABELS_PATH} not found — falling back to draft gold.")
    print(f"WARNING: {DRAFT_GOLD_WARNING}")
    records = load_draft_gold()
    texts, y_true, technique_names = load_gold_as_arrays(records)
    return texts, y_true, technique_names, True


def evaluate_on_true_gold(trainer: "Trainer", tokenizer, technique_names: list[str]) -> list:
    """The real held-out eval: the ~60 handcrafted_india messages, which
    were excluded from training entirely (see load_training_data). This is
    the number that belongs in the three-way baseline/teacher/student table
    — the Trainer's own eval_dataset (a random split of the trainable pool)
    is only a training-loop sanity check, not this.
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    records = load_draft_gold()
    texts = [r["text"] for r in records]
    mlb = MultiLabelBinarizer(classes=technique_names)
    y_true = mlb.fit_transform([r["labels"] for r in records])

    dataset = Dataset.from_dict({"text": texts, "labels": y_true.astype(np.float32).tolist()})
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256), batched=True
    )
    predictions = trainer.predict(dataset)
    proba = 1 / (1 + np.exp(-predictions.predictions))
    return per_technique_metrics(y_true, proba, technique_names)


def compute_metrics(eval_pred: EvalPrediction, technique_names: list[str]) -> dict:
    logits, labels = eval_pred
    proba = 1 / (1 + np.exp(-logits))  # sigmoid, since this is multi-label
    results = per_technique_metrics(labels, proba, technique_names)
    macro_f1 = float(np.mean([r.f1 for r in results]))
    return {"macro_f1": macro_f1}


def main() -> None:
    texts, y_true, technique_names, is_smoke_test = load_training_data()
    n = len(texts)

    if is_smoke_test:
        print(f"\n{'=' * 70}\nSMOKE TEST ONLY — N={n}, no silver-labeled corpus yet.\n{'=' * 70}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    split_at = max(1, int(n * 0.8))
    dataset = Dataset.from_dict({"text": texts, "labels": y_true.astype(np.float32).tolist()})
    dataset = dataset.train_test_split(train_size=split_at, seed=42)
    tokenized = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(technique_names),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=str(STUDENT_MODEL_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=lambda p: compute_metrics(p, technique_names),
    )

    trainer.train()

    predictions = trainer.predict(tokenized["test"])
    proba = 1 / (1 + np.exp(-predictions.predictions))
    y_true_test = np.array(tokenized["test"]["labels"])
    results = per_technique_metrics(y_true_test, proba, technique_names)

    title = (
        "STUDENT (ModernBERT-base) — SMOKE TEST"
        if is_smoke_test
        else "STUDENT (ModernBERT-base) — internal validation split (training-loop sanity check only)"
    )
    print(format_report(results, title))

    if not is_smoke_test:
        print(f"\n{'=' * 70}\nWARNING: {DRAFT_GOLD_WARNING}\n{'=' * 70}")
        gold_results = evaluate_on_true_gold(trainer, tokenizer, technique_names)
        print(format_report(gold_results, "STUDENT (ModernBERT-base) — TRUE GOLD HOLDOUT (never seen in training)"))

    STUDENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(STUDENT_MODEL_DIR))
    tokenizer.save_pretrained(str(STUDENT_MODEL_DIR))
    print(f"\nSaved student model to {STUDENT_MODEL_DIR}")


if __name__ == "__main__":
    main()
