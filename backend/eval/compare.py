"""Three-way comparison: TF-IDF baseline vs. Ollama teacher vs. distilled
student, all evaluated on the same held-out set — `eval.gold.load_real_gold()`
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
from app.student import STUDENT_MODEL_DIR
from eval.baseline import load_gold_as_arrays, run_baseline_cv
from eval.gold import GOLD_NOTE, load_real_gold
from eval.metrics import format_report, macro_average, per_technique_metrics, per_technique_metrics_cv


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
    """Returns (y_proba, avg_latency_seconds, n_params). Delegates the
    actual model loading + forward pass to app.student, the same module the
    live /analyze cascade uses — so this eval and the running gate are
    guaranteed to exercise identical inference, not two implementations
    that can drift apart."""
    from app.student import n_params as student_n_params
    from app.student import predict_proba_array

    probas = []
    total_elapsed = 0.0
    for text in texts:
        start = time.monotonic()
        probas.append(predict_proba_array(text))
        total_elapsed += time.monotonic() - start
    return np.array(probas), total_elapsed / len(texts), student_n_params()


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def _report_both(y_true, y_proba, technique_names: list[str], label: str) -> tuple[list, list]:
    """Prints the in-sample report (threshold tuned and scored on the same
    250 messages — an upper bound, see per_technique_metrics' docstring)
    followed by the held-out 5-fold CV report (threshold for each message
    chosen without seeing that message's label — the honest generalization
    estimate). Returns (in_sample_results, cv_results) so callers can build
    the summary from whichever they want.
    """
    in_sample = per_technique_metrics(y_true, y_proba, technique_names)
    print(format_report(in_sample, f"{label} — IN-SAMPLE (threshold tuned on the same set it's scored on; upper bound)"))

    cv = per_technique_metrics_cv(y_true, y_proba, technique_names)
    print(format_report(cv, f"{label} — HELD-OUT (5-fold threshold CV; genuine generalization estimate)"))

    return in_sample, cv


def main() -> None:
    records = load_real_gold()
    print("=" * 70)
    print(GOLD_NOTE)
    print(f"N = {len(records)}")
    print(
        "Every system below is reported two ways: IN-SAMPLE (thresholds tuned "
        "and scored on the same 250 messages — an upper bound) and HELD-OUT "
        "(5-fold threshold CV — the genuine estimate). See ERROR_ANALYSIS.md."
    )
    print("=" * 70)

    texts, y_true, technique_names = load_gold_as_arrays(records)

    baseline_proba = run_baseline_cv(texts, y_true)
    _, baseline_cv = _report_both(y_true, baseline_proba, technique_names, "BASELINE: TF-IDF + Logistic Regression")

    teacher_proba, teacher_latency, teacher_errors = teacher_predict(texts, technique_names)
    _, teacher_cv = _report_both(y_true, teacher_proba, technique_names, f"TEACHER: {TEACHER_MODEL} (Ollama, live)")
    if teacher_errors:
        print(f"  ({teacher_errors}/{len(texts)} gold messages errored on the teacher and count as all-zero predictions)")

    student_cv = None
    student_latency = student_size_mb = student_params = None
    if STUDENT_MODEL_DIR.exists():
        student_proba, student_latency, student_params = student_predict(texts)
        _, student_cv = _report_both(y_true, student_proba, technique_names, "STUDENT: distilled ModernBERT-base")
        student_size_mb = dir_size_mb(STUDENT_MODEL_DIR)
    else:
        print(f"\nNo student model at {STUDENT_MODEL_DIR} - run eval/distill.py first. Skipping student row.")

    print("\n" + "=" * 70)
    print("SUMMARY - macro-F1 (held-out, 5-fold threshold CV), latency/message, model size")
    print("=" * 70)
    b_macro = macro_average(baseline_cv)
    t_macro = macro_average(teacher_cv)
    print(f"{'BASELINE (TF-IDF+LR)':<32} macro-F1={b_macro['f1']:.2f}  latency=~0ms (CPU, in-process)  size=negligible")
    print(
        f"{'TEACHER (' + TEACHER_MODEL + ')':<32} macro-F1={t_macro['f1']:.2f}  "
        f"latency={teacher_latency * 1000:.0f}ms/msg  size=~4.5GB (Q4_K_M gguf)"
    )
    if student_cv:
        s_macro = macro_average(student_cv)
        print(
            f"{'STUDENT (ModernBERT-base)':<32} macro-F1={s_macro['f1']:.2f}  "
            f"latency={student_latency * 1000:.0f}ms/msg  size={student_size_mb:.0f}MB ({student_params / 1e6:.0f}M params)"
        )


if __name__ == "__main__":
    main()
