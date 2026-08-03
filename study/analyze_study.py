"""Fixed analysis plan from PREREGISTRATION.md, run against
`data/responses.jsonl`.

Primary: difference in proportion resistant (treatment - control), pooled
across participants and the 4 scam stimuli, with a bootstrap 95% CI.
Secondary: the same comparison on the 2 legitimate stimuli — a positive
difference there (treatment participants trusting a real message LESS
than control) is the H2 harm signal and is surfaced prominently, not
buried.

No per-stimulus significance testing and no subgroup analysis — the
pre-registration is explicit that the six stimuli are not six separate
experiments. A purely descriptive (non-inferential) per-stimulus count
table is printed for error-analysis purposes only, clearly labeled as such.

Run: python study/analyze_study.py
"""

import json
import math
import random
import sys
from pathlib import Path

RESPONSES_PATH = Path(__file__).parent / "data" / "responses.jsonl"

RESISTANT = {"ignore", "verify_independently", "call_someone"}
NON_RESISTANT = {"comply", "reply"}


def load_responses(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} not found. Run study/run_study.py first to collect responses.")
        sys.exit(1)
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_resistant(response_code: str) -> int:
    if response_code in RESISTANT:
        return 1
    if response_code in NON_RESISTANT:
        return 0
    raise ValueError(f"unrecognized response code: {response_code!r}")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a single proportion — appropriate at small
    n, unlike a normal approximation (see prereg 'Analysis plan')."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_diff_ci(
    control_outcomes: list[int],
    treatment_outcomes: list[int],
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Returns (point_estimate, lo, hi) for treatment_mean - control_mean."""
    if not control_outcomes or not treatment_outcomes:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    point = sum(treatment_outcomes) / len(treatment_outcomes) - sum(control_outcomes) / len(control_outcomes)

    diffs = []
    for _ in range(n_boot):
        c_sample = [rng.choice(control_outcomes) for _ in control_outcomes]
        t_sample = [rng.choice(treatment_outcomes) for _ in treatment_outcomes]
        diffs.append(sum(t_sample) / len(t_sample) - sum(c_sample) / len(c_sample))
    diffs.sort()
    lo = diffs[int(alpha / 2 * n_boot)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (point, lo, hi)


def summarize_arm(outcomes: list[int]) -> str:
    n = len(outcomes)
    successes = sum(outcomes)
    prop = successes / n if n else 0.0
    lo, hi = wilson_interval(successes, n)
    return f"{successes}/{n} = {prop:.2f}  (95% Wilson CI [{lo:.2f}, {hi:.2f}])"


def report_comparison(title: str, control: list[int], treatment: list[int]) -> None:
    print(f"\n{title}")
    print(f"  Control:   {summarize_arm(control)}")
    print(f"  Treatment: {summarize_arm(treatment)}")
    point, lo, hi = bootstrap_diff_ci(control, treatment)
    print(f"  Difference (treatment - control): {point:+.2f}  (95% bootstrap CI [{lo:+.2f}, {hi:+.2f}])")


def main() -> None:
    records = load_responses(RESPONSES_PATH)
    n_participants = len({r["participant_id"] for r in records})
    print(f"Loaded {len(records)} responses from {n_participants} participants.")

    scam = [r for r in records if r["kind"] == "scam"]
    legit = [r for r in records if r["kind"] == "legitimate"]

    scam_control = [is_resistant(r["response"]) for r in scam if r["arm"] == "control"]
    scam_treatment = [is_resistant(r["response"]) for r in scam if r["arm"] == "treatment"]
    legit_control = [is_resistant(r["response"]) for r in legit if r["arm"] == "control"]
    legit_treatment = [is_resistant(r["response"]) for r in legit if r["arm"] == "treatment"]

    report_comparison("PRIMARY (H1) — proportion resistant on 4 scam stimuli:", scam_control, scam_treatment)

    report_comparison(
        "SECONDARY (H2) — proportion resistant/distrustful on 2 legitimate stimuli:", legit_control, legit_treatment
    )
    legit_point, _, _ = bootstrap_diff_ci(legit_control, legit_treatment)
    if legit_point > 0:
        print(
            "  NOTE: treatment shows a HIGHER false-alarm rate than control on legitimate "
            "messages. This is the H2 harm signal — report it, do not bury it."
        )

    print("\nDescriptive per-stimulus counts (NOT a statistical comparison — see prereg 'no subgroup analysis'):")
    stimulus_ids = sorted({r["stimulus_id"] for r in records})
    for sid in stimulus_ids:
        for arm in ("control", "treatment"):
            outcomes = [is_resistant(r["response"]) for r in records if r["stimulus_id"] == sid and r["arm"] == arm]
            if outcomes:
                print(f"  {sid:28s} {arm:10s} resistant={sum(outcomes)}/{len(outcomes)}")


if __name__ == "__main__":
    main()
