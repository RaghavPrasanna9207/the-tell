"""Three-way comparison: TF-IDF baseline vs. Ollama teacher vs. distilled
student, all evaluated on the same held-out set — `eval.gold.load_draft_gold()`
— so the numbers are directly comparable. Per-technique + macro metrics,
latency, and model size (see plan Week 2, step 4).

Requires Ollama running (for the teacher) and a trained student model at
`models/student` (run `eval/distill.py` first — this script degrades
gracefully and just skips the student row if that model doesn't exist).

Run: python eval/compare.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.classify import classify
from app.llm import TEACHER_MODEL, check_available
from eval.baseline import load_gold_as_arrays, run_baseline_cv
from eval.gold import DRAFT_GOLD_WARNING, load_draft_gold
from eval.metrics import format_report, macro_average, per_technique_metrics

STUDENT_MODEL_DIR = Path(__file__).parent.parent / "models" / "student"


def teacher_predict(texts: list[str], technique_names: list[str]) -> tuple[np.ndarray, float, int]:
    """Runs the real teacher (classify()) on each text. Returns (y_proba,
    avg_latency_seconds, n_errors). Confidence is used as the "probability"
    for detected techniques, 0.0 otherwise — a point estimate, not a
    calibrated probability (same caveat as the live product).

    A message that fails Pydantic validation (seen in practice: the model
    occasionally emits a confidence > 1.0 — schema-constrained decoding
    enforces JSON type, not numeric range) is counted as an error and left
    as all-zero rather than aborting the whole comparison run, matching
    eval/silver_label.py's error handling."""
    check_available()
    name_to_idx = {name: i for i, name in enumerate(technique_names)}
    y_proba = np.zeros((len(texts), len(technique_names)))
    total_elapsed = 0.0
    n_errors = 0
    for i, text in enumerate(texts):
        try:
            result = classify(text)
        except Exception as exc:
            n_errors += 1
            print(f"  ERROR on gold message {i}: {exc}")
            continue
        total_elapsed += result.generation.elapsed_seconds
        for d in result.analysis.detections:
            y_proba[i, name_to_idx[d.technique.value]] = d.confidence
    n_ok = len(texts) - n_errors
    return y_proba, (total_elapsed / n_ok if n_ok else 0.0), n_errors


def student_predict(texts: list[str]) -> tuple[np.ndarray, float, int]:
    """Returns (y_proba, avg_latency_seconds, n_params). `datasets` is never
    imported in this process, so the torch-import-order segfault documented
    in distill.py doesn't apply here — torch can be imported directly."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(STUDENT_MODEL_DIR))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    probas = []
    total_elapsed = 0.0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            start = time.monotonic()
            logits = model(**inputs).logits
            total_elapsed += time.monotonic() - start
            probas.append(torch.sigmoid(logits).cpu().numpy()[0])
    return np.array(probas), total_elapsed / len(texts), n_params


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def main() -> None:
    records = load_draft_gold()
    print("=" * 70)
    print(f"WARNING: {DRAFT_GOLD_WARNING}")
    print(f"N = {len(records)}")
    print("=" * 70)

    texts, y_true, technique_names = load_gold_as_arrays(records)

    baseline_proba = run_baseline_cv(texts, y_true)
    baseline_results = per_technique_metrics(y_true, baseline_proba, technique_names)
    print(format_report(baseline_results, "BASELINE: TF-IDF + Logistic Regression (5-fold CV)"))

    teacher_proba, teacher_latency, teacher_errors = teacher_predict(texts, technique_names)
    teacher_results = per_technique_metrics(y_true, teacher_proba, technique_names)
    print(format_report(teacher_results, f"TEACHER: {TEACHER_MODEL} (Ollama, live)"))
    if teacher_errors:
        print(f"  ({teacher_errors}/{len(texts)} gold messages errored on the teacher and count as all-zero predictions)")

    student_results = None
    student_latency = student_size_mb = student_params = None
    if STUDENT_MODEL_DIR.exists():
        student_proba, student_latency, student_params = student_predict(texts)
        student_results = per_technique_metrics(y_true, student_proba, technique_names)
        student_size_mb = dir_size_mb(STUDENT_MODEL_DIR)
        print(format_report(student_results, "STUDENT: distilled ModernBERT-base"))
    else:
        print(f"\nNo student model at {STUDENT_MODEL_DIR} - run eval/distill.py first. Skipping student row.")

    print("\n" + "=" * 70)
    print("SUMMARY - macro-F1, latency/message, model size")
    print("=" * 70)
    b_macro = macro_average(baseline_results)
    t_macro = macro_average(teacher_results)
    print(f"{'BASELINE (TF-IDF+LR)':<32} macro-F1={b_macro['f1']:.2f}  latency=~0ms (CPU, in-process)  size=negligible")
    print(
        f"{'TEACHER (' + TEACHER_MODEL + ')':<32} macro-F1={t_macro['f1']:.2f}  "
        f"latency={teacher_latency * 1000:.0f}ms/msg  size=~4.5GB (Q4_K_M gguf)"
    )
    if student_results:
        s_macro = macro_average(student_results)
        print(
            f"{'STUDENT (ModernBERT-base)':<32} macro-F1={s_macro['f1']:.2f}  "
            f"latency={student_latency * 1000:.0f}ms/msg  size={student_size_mb:.0f}MB ({student_params / 1e6:.0f}M params)"
        )


if __name__ == "__main__":
    main()
