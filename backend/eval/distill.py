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
STUDENT_MODEL_DIR = Path(__file__).parent.parent / "models" / "student"


def load_training_data() -> tuple[list[str], np.ndarray, list[str], bool]:
    """Returns (texts, y_true, technique_names, is_smoke_test)."""
    if SILVER_LABELS_PATH.exists():
        records = []
        with SILVER_LABELS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        texts = [r["text"] for r in records]
        technique_names = [t.value for t in Technique]
        from sklearn.preprocessing import MultiLabelBinarizer

        mlb = MultiLabelBinarizer(classes=technique_names)
        y_true = mlb.fit_transform([r["labels"] for r in records])
        return texts, y_true, technique_names, False

    print(f"WARNING: {SILVER_LABELS_PATH} not found — falling back to draft gold.")
    print(f"WARNING: {DRAFT_GOLD_WARNING}")
    records = load_draft_gold()
    texts, y_true, technique_names = load_gold_as_arrays(records)
    return texts, y_true, technique_names, True


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

    title = "STUDENT (ModernBERT-base) — SMOKE TEST" if is_smoke_test else "STUDENT (ModernBERT-base)"
    print(format_report(results, title))

    STUDENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(STUDENT_MODEL_DIR))
    tokenizer.save_pretrained(str(STUDENT_MODEL_DIR))
    print(f"\nSaved student model to {STUDENT_MODEL_DIR}")


if __name__ == "__main__":
    main()
