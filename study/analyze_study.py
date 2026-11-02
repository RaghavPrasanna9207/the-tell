"""Fixed analysis plan from PREREGISTRATION.md, run against
`data/responses.jsonl`.

Primary (H1): difference in proportion resistant (treatment - control), pooled
across participants and the 4 scam stimuli, with a bootstrap 95% CI. This
stands as pre-registered.

Secondary (H2): **WITHDRAWN as uninterpretable** — see the 2026-09-08 amendment
in PREREGISTRATION.md. Two instrument defects, both in the design rather than
in the data or this analysis:

  1. The two arms saw the same screen on the only two stimuli H2 uses. S5/S6
     classified clean, so control got "[No warning was flagged...]" and
     treatment got "No manipulation techniques detected." — undifferentiated.
  2. `ignore` is coded as "distrust", but ignoring a delivery notification is
     the correct ordinary response, not a false alarm. Nearly all of the
     "distrust" signal on the legitimate stimuli is exactly that response.

The H2 numbers are still computed and printed, because deleting them would
hide the error rather than report it — but they print under an explicit
UNINTERPRETABLE header carrying both defects and the ignore-share diagnostic,
so the number cannot travel without its caveat.

`report_genuineness()` implements the corrected measure (a direct "is this
message genuine?" judgment, which separates the two things the resistant
coding fuses) for any future run. The existing 120 responses predate the field
and are skipped.

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


def load_from_aggregates(path: Path) -> tuple[list[dict], int]:
    """Rebuild an analysable record list from committed aggregate counts.

    `study/data/` is gitignored (PREREGISTRATION.md, "Data handling"), so a
    fresh clone has no responses.jsonl and previously could not reproduce
    Eval B at all despite the README saying it could. `aggregate_counts.json`
    holds per-stimulus x per-arm counts with no ids and no linkage, which is
    enough to recompute every number here: the outcomes are binary, so a
    (k of n) count fully determines the multiset the analysis operates on.

    One caveat, stated rather than hidden: the bootstrap resamples from a
    list, so reconstructed order is not the original order and the CI bounds
    can differ from the responses.jsonl run in the last decimal. The point
    estimates and the Wilson intervals are exact.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for cell in data["cells"]:
        for response_code, count in sorted(cell["responses"].items()):
            for _ in range(count):
                records.append(
                    {
                        "participant_id": None,
                        "arm": cell["arm"],
                        "stimulus_id": cell["stimulus_id"],
                        "kind": cell["kind"],
                        "response": response_code,
                        "believes_genuine": None,
                    }
                )
    return records, data["_n_participants"]


def load_responses(path: Path) -> tuple[list[dict], int]:
    if not path.exists():
        aggregate_path = path.parent / "aggregate_counts.json"
        if aggregate_path.exists():
            records, n_participants = load_from_aggregates(aggregate_path)
            print(f"NOTE: {path.name} not found — running from committed aggregate counts")
            print("      instead. Point estimates and Wilson CIs are exact; bootstrap CI")
            print("      bounds may differ in the last decimal (see load_from_aggregates).")
            return records, n_participants
        print(f"ERROR: {path} not found. Run study/run_study.py first to collect responses.")
        sys.exit(1)
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records, len({r["participant_id"] for r in records})


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


def ignore_share(records: list[dict], arm: str) -> tuple[int, int]:
    """Of the responses counted as 'resistant' in this arm, how many are
    literally `ignore`? This is the H2 defect made numeric: on a legitimate
    informational message, `ignore` means "nothing to do here", not "I distrust
    this" — so a resistant count made almost entirely of `ignore` is not
    measuring distrust at all. Returns (n_ignore, n_resistant)."""
    arm_records = [r for r in records if r["arm"] == arm]
    resistant = [r for r in arm_records if r["response"] in RESISTANT]
    return sum(1 for r in resistant if r["response"] == "ignore"), len(resistant)


def report_h2_withdrawn(legit: list[dict], control: list[int], treatment: list[int]) -> None:
    """Print H2's numbers under the defects that make them uninterpretable.

    The numbers are printed rather than suppressed on purpose: this analysis
    was reported publicly before the defects were found, so the correction has
    to show the original number next to what is wrong with it. Hiding it would
    make the record less honest, not more.
    """
    print("\n" + "!" * 72)
    print("SECONDARY (H2) — WITHDRAWN AS UNINTERPRETABLE")
    print("!" * 72)
    report_comparison(
        "  [computed anyway, for the record] proportion 'resistant' on 2 legitimate stimuli:",
        control,
        treatment,
    )
    print("\n  Why this number does not mean what it appears to mean:")
    print("\n  DEFECT 1 — the arms were not differentiated on these two stimuli.")
    print("    S5 and S6 both classified clean, so control saw")
    print('      "[No warning was flagged for this message.]"')
    print("    and treatment saw")
    print('      "No manipulation techniques detected."')
    print("    Same screen, different wording. The only live mechanism is carryover")
    print("    from the 4 explanations seen earlier — a much weaker test than the")
    print("    pre-registration described, and never stated as the mechanism.")
    print("\n  DEFECT 2 — 'ignore' is being counted as distrust.")
    for arm, outcomes in (("control", control), ("treatment", treatment)):
        n_ign, n_res = ignore_share(legit, arm)
        pct = f"{n_ign / n_res:.0%}" if n_res else "n/a"
        print(f"    {arm:<10} {n_ign}/{n_res} of 'distrustful' responses are literally `ignore`  ({pct})")
    print("    Ignoring a delivery notification is correct behavior, not a false")
    print("    alarm. This measures 'did you decline to act on an informational")
    print("    SMS', not over-distrust.")
    print("\n  => H2 is not a null result. It is a measurement that could not have")
    print("     detected the harm it claimed to rule out. Do not cite it as evidence")
    print("     of no harm. See PREREGISTRATION.md, amendment 2026-09-08.")
    print("     The corrected instrument is below.")


def report_genuineness(records: list[dict]) -> None:
    """The corrected H2 measure: a direct judgment ('is this message genuine?')
    rather than one inferred from a chosen action. On a legitimate stimulus,
    answering 'not genuine' is a false alarm no matter what action the
    participant would take — which is exactly the separation the resistant
    coding fuses. Silent when no run has collected the field yet."""
    scored = [r for r in records if r.get("believes_genuine") is not None]
    if not scored:
        return
    print("\nCORRECTED H2 — proportion judging a LEGITIMATE message 'not genuine' (false alarm):")
    legit = [r for r in scored if r["kind"] == "legitimate"]
    control = [0 if r["believes_genuine"] else 1 for r in legit if r["arm"] == "control"]
    treatment = [0 if r["believes_genuine"] else 1 for r in legit if r["arm"] == "treatment"]
    if control and treatment:
        report_comparison("  False-alarm rate on legitimate stimuli:", control, treatment)
        point, _, _ = bootstrap_diff_ci(control, treatment)
        if point > 0:
            print("  NOTE: treatment false-alarms MORE than control. This is the H2 harm")
            print("        signal on the corrected measure — report it, do not bury it.")


def report_comparison(title: str, control: list[int], treatment: list[int]) -> None:
    print(f"\n{title}")
    print(f"  Control:   {summarize_arm(control)}")
    print(f"  Treatment: {summarize_arm(treatment)}")
    point, lo, hi = bootstrap_diff_ci(control, treatment)
    print(f"  Difference (treatment - control): {point:+.2f}  (95% bootstrap CI [{lo:+.2f}, {hi:+.2f}])")


def main() -> None:
    records, n_participants = load_responses(RESPONSES_PATH)
    print(f"Loaded {len(records)} responses from {n_participants} participants.")

    scam = [r for r in records if r["kind"] == "scam"]
    legit = [r for r in records if r["kind"] == "legitimate"]

    scam_control = [is_resistant(r["response"]) for r in scam if r["arm"] == "control"]
    scam_treatment = [is_resistant(r["response"]) for r in scam if r["arm"] == "treatment"]
    legit_control = [is_resistant(r["response"]) for r in legit if r["arm"] == "control"]
    legit_treatment = [is_resistant(r["response"]) for r in legit if r["arm"] == "treatment"]

    report_comparison("PRIMARY (H1) — proportion resistant on 4 scam stimuli:", scam_control, scam_treatment)

    report_h2_withdrawn(legit, legit_control, legit_treatment)
    report_genuineness(records)

    print("\nDescriptive per-stimulus counts (NOT a statistical comparison — see prereg 'no subgroup analysis'):")
    stimulus_ids = sorted({r["stimulus_id"] for r in records})
    for sid in stimulus_ids:
        for arm in ("control", "treatment"):
            outcomes = [is_resistant(r["response"]) for r in records if r["stimulus_id"] == sid and r["arm"] == arm]
            if outcomes:
                print(f"  {sid:28s} {arm:10s} resistant={sum(outcomes)}/{len(outcomes)}")


if __name__ == "__main__":
    main()
