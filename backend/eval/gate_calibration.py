"""Calibrates and reports the Phase-A1 cascade gate's threshold trade-off:
how much scam recall the student gate buys/costs at a given probability
cutoff, and how often it needlessly wakes the teacher on clean messages.
The teacher itself is never called here — this only needs the student's
already-fast probabilities against the real gold set, so it's cheap to
rerun whenever the student is retrained.

Run: python eval/gate_calibration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.student import GATE_THRESHOLD, STUDENT_MODEL_DIR
from eval.baseline import load_gold_as_arrays
from eval.compare import student_predict
from eval.gold import GOLD_NOTE, load_real_gold

CANDIDATE_THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]


def main() -> None:
    if not STUDENT_MODEL_DIR.exists():
        print(f"No student model at {STUDENT_MODEL_DIR} — run eval/distill.py first.")
        return

    records = load_real_gold()
    texts, y_true, technique_names = load_gold_as_arrays(records)
    y_proba, avg_latency, params = student_predict(texts)

    is_scam = y_true.sum(axis=1) > 0
    n_scam = int(is_scam.sum())
    n_clean = len(texts) - n_scam

    print("=" * 70)
    print(GOLD_NOTE)
    print(f"N = {len(texts)} ({n_scam} scam, {n_clean} clean)")
    print(f"Student: {params / 1e6:.0f}M params, {avg_latency * 1000:.1f}ms/msg avg")
    print("=" * 70)
    print(
        "\nGate fires if ANY technique's student probability >= threshold.\n"
        "'scam recall'       = fraction of scam messages that wake the teacher "
        "(a miss here means Layer 2 never runs for that message at all).\n"
        "'clean wake rate'   = fraction of clean messages that needlessly wake "
        "the teacher (a latency cost only — the teacher would presumably say "
        "'nothing' too, so this doesn't cost correctness).\n"
        "'overall wake rate' = fraction of ALL gold messages that reach the "
        "teacher — the actual latency saving the cascade buys.\n"
    )
    print(f"{'threshold':>10} {'scam recall':>14} {'clean wake rate':>17} {'overall wake rate':>19}")
    max_proba = y_proba.max(axis=1)
    for threshold in CANDIDATE_THRESHOLDS:
        fires = max_proba >= threshold
        scam_recall = fires[is_scam].mean() if n_scam else float("nan")
        clean_wake_rate = fires[~is_scam].mean() if n_clean else float("nan")
        overall_wake_rate = fires.mean()
        marker = "  <- GATE_THRESHOLD (app/student.py)" if abs(threshold - GATE_THRESHOLD) < 1e-9 else ""
        print(f"{threshold:>10.2f} {scam_recall:>14.2f} {clean_wake_rate:>17.2f} {overall_wake_rate:>19.2f}{marker}")

    print(
        "\nCompare 'scam recall' at GATE_THRESHOLD against the teacher's own "
        "held-out macro-recall (eval/compare.py) — the gap between them is "
        "the additional recall the gate itself costs, on top of whatever the "
        "teacher already misses once it's actually called."
    )


if __name__ == "__main__":
    main()
