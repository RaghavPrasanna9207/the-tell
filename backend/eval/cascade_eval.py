"""End-to-end eval of the system that actually ships.

`eval/compare.py` scores the teacher as if it were called on every message.
It isn't. `app/main.py` puts the distilled student in front of it as a
high-recall gate, and a message the gate rejects returns "no manipulation
techniques detected" with the teacher never running. So the headline 0.51
macro-F1 describes a component, not the product — the deployed pipeline is
strictly worse than its teacher, and by how much was never measured.

`eval/gate_calibration.py` measures the gate alone ("does anything fire",
message-level). This composes the two and scores the result the same way
`compare.py` scores everything else, so the number is directly comparable.

Design note — why one teacher pass, scored twice: the teacher is not
deterministic call-to-call even at temperature 0.1 (see ERROR_ANALYSIS.md's
note on a ~0.01 macro-F1 wobble between runs). Running it once over all 250
messages and then scoring (a) every prediction and (b) predictions with
gate-rejected rows zeroed out means the ONLY difference between the two
numbers is the gate. Running two separate teacher passes would confound the
gate's cost with sampling noise.

Cost: one full teacher pass (~250 calls). Cached to .cache/ so re-scoring is
free; use --refresh to force a new pass.

Run: python eval/cascade_eval.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.llm import TEACHER_MODEL, check_available
from app.student import GATE_THRESHOLD, STUDENT_MODEL_DIR, should_investigate
from eval.baseline import load_gold_as_arrays
from eval.compare import teacher_predict
from eval.gold import GOLD_NOTE, load_real_gold
from eval.metrics import format_report, macro_average, per_technique_metrics_cv

CACHE_PATH = Path(__file__).parent / ".cache" / "teacher_gold_proba.json"


def load_or_run_teacher(texts: list[str], technique_names: list[str], refresh: bool):
    """Teacher probabilities over every gold message, cached. The cache is
    keyed on the message list so a changed gold set can't silently reuse
    stale predictions."""
    key = str(len(texts)) + "|" + (texts[0][:50] if texts else "")
    if CACHE_PATH.exists() and not refresh:
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if cached.get("key") == key and cached.get("model") == TEACHER_MODEL:
            print(f"Using cached teacher pass ({CACHE_PATH}). --refresh to re-run.")
            return np.array(cached["y_proba"]), cached["avg_latency"], cached["n_errors"]
        print("Cache exists but doesn't match the current gold set/model — re-running.")

    check_available()
    print(f"Running teacher ({TEACHER_MODEL}) over {len(texts)} messages. This is the slow part.")
    y_proba, avg_latency, n_errors = teacher_predict(texts, technique_names)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(
            {
                "key": key,
                "model": TEACHER_MODEL,
                "y_proba": y_proba.tolist(),
                "avg_latency": avg_latency,
                "n_errors": n_errors,
            }
        ),
        encoding="utf-8",
    )
    return y_proba, avg_latency, n_errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore the cached teacher pass")
    args = ap.parse_args()

    if not STUDENT_MODEL_DIR.exists():
        print(f"No student model at {STUDENT_MODEL_DIR} — run eval/distill.py first.")
        print("(Without a student there is no gate, so the cascade IS the teacher.)")
        return

    records = load_real_gold()
    texts, y_true, technique_names = load_gold_as_arrays(records)

    print("=" * 72)
    print(GOLD_NOTE)
    print("=" * 72)

    # --- the gate, using the exact predicate app/main.py calls -------------
    gate_start = time.monotonic()
    gate_open = np.array([should_investigate(t) for t in texts])
    gate_elapsed = time.monotonic() - gate_start

    is_scam = y_true.sum(axis=1) > 0
    killed = ~gate_open & is_scam
    n_scam = int(is_scam.sum())

    print(f"\nGate (threshold {GATE_THRESHOLD}, app/student.py):")
    print(f"  wakes the teacher for {gate_open.sum()}/{len(texts)} messages ({gate_open.mean():.0%})")
    print(f"  scam recall {gate_open[is_scam].mean():.2f}  ({n_scam - killed.sum()}/{n_scam} scam messages get through)")
    print(f"  gate latency {gate_elapsed / len(texts) * 1000:.0f}ms/msg")

    # --- one teacher pass, cached ------------------------------------------
    y_proba_teacher, teacher_latency, n_errors = load_or_run_teacher(texts, technique_names, args.refresh)
    if n_errors:
        print(f"  ({n_errors} messages errored during the teacher pass — counted as all-zero)")

    # --- the same predictions, as the product would actually return them ---
    # A gate-rejected message returns is_clean=True with no cards. That is an
    # all-zero prediction row, not a missing one: the product gave an answer,
    # it was just "nothing here".
    y_proba_cascade = y_proba_teacher.copy()
    y_proba_cascade[~gate_open] = 0.0

    teacher_results = per_technique_metrics_cv(y_true, y_proba_teacher, technique_names)
    cascade_results = per_technique_metrics_cv(y_true, y_proba_cascade, technique_names)

    print(format_report(teacher_results, "TEACHER ALONE (held-out CV) — what compare.py reports"))
    print(format_report(cascade_results, "DEPLOYED CASCADE (held-out CV) — student gate -> teacher"))

    t_macro = macro_average(teacher_results)
    c_macro = macro_average(cascade_results)

    print("\n" + "=" * 72)
    print("WHAT THE GATE COSTS")
    print("=" * 72)
    print(f"{'':<22} {'macro-P':>9} {'macro-R':>9} {'macro-F1':>9}")
    print(f"{'teacher alone':<22} {t_macro['precision']:>9.2f} {t_macro['recall']:>9.2f} {t_macro['f1']:>9.2f}")
    print(f"{'deployed cascade':<22} {c_macro['precision']:>9.2f} {c_macro['recall']:>9.2f} {c_macro['f1']:>9.2f}")
    print(
        f"{'difference':<22} {c_macro['precision'] - t_macro['precision']:>+9.2f} "
        f"{c_macro['recall'] - t_macro['recall']:>+9.2f} {c_macro['f1'] - t_macro['f1']:>+9.2f}"
    )

    # Latency: the whole point of the cascade. Gate runs on everything; the
    # teacher only on what gets through.
    gate_ms = gate_elapsed / len(texts) * 1000
    blended_ms = gate_ms + gate_open.mean() * teacher_latency * 1000
    print(f"\nLatency/msg: teacher-always {teacher_latency * 1000:.0f}ms  ->  cascade {blended_ms:.0f}ms")
    print(f"  ({gate_ms:.0f}ms gate on every message + {gate_open.mean():.0%} of messages paying the teacher's")
    print(f"   {teacher_latency * 1000:.0f}ms). The {1 - gate_open.mean():.0%} that never reach the teacher are the saving.")

    print("\n" + "=" * 72)
    print(f"SCAM MESSAGES THE GATE SILENCED ({int(killed.sum())} of {n_scam})")
    print("=" * 72)
    print("For these, /analyze returns 'no manipulation techniques detected' and")
    print("Layer 2 never runs. This is the cascade's real correctness cost and the")
    print("most product-relevant number here, for a tool whose rule is never to")
    print("render 'safe'.\n")
    for idx in np.where(killed)[0]:
        rec = records[idx]
        print(f"  {rec['id']:<12} {','.join(rec['labels']):<45} {rec['text'][:60]!r}")
    if not killed.any():
        print("  (none — the gate let every scam message through on this gold set)")


if __name__ == "__main__":
    main()
