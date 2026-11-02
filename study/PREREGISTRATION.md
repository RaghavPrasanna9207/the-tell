# Pre-registration: Does naming the manipulation technique change stated behavior?

**Committed before recruiting any participant.** This document is not to be edited
after the first response is collected — amendments after that point must be logged
in a dated addendum below, not silently merged into the sections above.

## Thesis

Detection doesn't change behavior. Telling someone "this looks like a scam" is a
verdict; naming the specific manipulation technique being used on them, grounded in
a falsifiable fact, is a counter-argument. The prebunking/inoculation literature
(van der Linden & Roozenbeek; *ShieldUp!*, arXiv 2503.12341) predicts the latter
should outperform the former on resistance to a live persuasion attempt. This study
tests that prediction directly, on this project's own explanations, rather than
citing the literature and assuming it transfers.

## Hypothesis

**H1:** Participants shown a named-technique explanation (treatment) will report a
more resistant stated action (verify-independently / call someone / ignore, rather
than comply or reply) on scam stimuli than participants shown a plain binary flag
(control).

**H2 (secondary, and the one we're least sure about):** The treatment will not
increase false-alarm behavior on legitimate messages relative to control. If it
does, that's a real cost of the explanation approach and gets reported as such —
this study is not designed to only look for the effect we want.

## Design

- **Between-subjects, 2 arms**, randomly assigned per participant.
  - **Control:** binary flag only — "⚠️ This looks like a scam."
  - **Treatment:** the full named-technique explanation (Layer 2 output).
- **N ≈ 20** total (~10 per arm). This is explicitly underpowered for a
  conventional significance threshold — the honest framing in the writeup is
  effect size + confidence interval, not a p-value verdict. Recruiting more than
  ~20 is out of scope for this project; if the CI is wide, that gets reported as
  a limitation, not hidden.

## Stimuli

Six messages shown to every participant, same order, same six messages regardless
of arm (only the flag/explanation differs):

1. Digital-arrest script (false authority + isolation + fear of consequence)
2. UPI "collect request" payment-irreversibility scam
3. Family-member impersonation ("Mom, lost my phone")
4. Lottery/prize reciprocity-hook scam
5. **Legitimate bank OTP message** (benign — tests false-alarm rate)
6. **Legitimate delivery/appointment notification** (benign — tests false-alarm rate)

Stimuli 5 and 6 are not filler. A design that only measures the wanted effect on
scam messages and never checks the cost on legitimate ones is not a real eval.

## Primary outcome

For each message, forced choice: **Comply / Reply to ask a question / Ignore /
Verify independently / Call someone.** Coded as resistant (verify independently,
call someone, ignore) vs. non-resistant (comply, reply) for the primary analysis.

## Secondary outcomes

- Self-reported confidence in the judgment (1-5 Likert).
- "Would you warn someone else about this message?" (yes/no).

## Analysis plan (fixed in advance)

- Primary: difference in proportion resistant (treatment - control) on the 4 scam
  stimuli, with a 95% CI (bootstrap or Wilson interval, not a normal approximation
  given the small N).
- Secondary: same proportion-resistant comparison on the 2 benign stimuli — a
  **positive** difference here (treatment participants distrusting a real bank SMS
  more than control) is the harm signal from H2 and must be reported prominently if
  present, not buried.
- No subgroup analysis, no p-hacking across the six stimuli individually as if they
  were six separate experiments — the primary claim is about the 4-scam aggregate.

## What would change our mind

If the treatment shows no detectable difference from control on the primary
outcome, or increases the false-alarm rate on benign stimuli, that is reported as
the finding. "Being right isn't the same as being believed" is only an honest
sentence to say afterward if it was a real possible outcome beforehand, not a
conclusion this study was designed to reach regardless of the data.

## Data handling

No raw message logs. Responses recorded as: participant id (random, not linked to
identity), arm, stimulus id, response code, confidence, timestamp. See docs/DESIGN_RULES.md —
no PII is collected as part of this study; participants are not asked to submit
their own real messages.

---

## Amendments

### 2026-09-08 — H2 is withdrawn as uninterpretable (two instrument defects)

Found during a post-hoc audit of the collected data, after all 20 participants had
run. **H2's result is withdrawn.** It is not a null finding; it is a measurement
that could not have detected the thing it claimed to measure. Both defects are in
the instrument this document specified, not in the data collection or the analysis
code, so they are logged here rather than fixed silently.

**Defect 1 — the two arms were not differentiated on the H2 stimuli.**
H2 is computed entirely from S5 (bank OTP) and S6 (delivery notification). The
pipeline classified both as clean (`is_clean=True` in `stimuli_cache.json`), so
`run_study.py` rendered control as `[No warning was flagged for this message.]`
and treatment as `No manipulation techniques detected.` Those are the same screen
in different words. The design section above assumed the treatment arm would show
"the full named-technique explanation" on every stimulus; on the only two stimuli
H2 uses, there was no explanation to show, because the system correctly found
nothing.

The one live mechanism that remains is carryover: treatment participants saw four
named-technique explanations on S1-S4 before reaching S5-S6. That is a legitimate
design, but it is a substantially weaker test than this document described, and it
was never stated as the mechanism. It is not what the analysis plan was written
against.

**Defect 2 — the outcome coding conflates "no action needed" with "distrust."**
The primary outcome codes ignore / verify_independently / call_someone as
"resistant." On a *scam* stimulus that coding is sound. On a *legitimate*
informational message it is not: `ignore` is the correct, ordinary response to a
delivery notification or an OTP you did not request — there is nothing to comply
with. In the collected data, of the responses counted as "distrustful" on the
legitimate stimuli, 15 of 17 (control) and **16 of 16** (treatment) are literally
`ignore`. The reported 0.85 / 0.80 "false-alarm rate" therefore measures "did you
decline to act on an informational SMS," not over-distrust. An 85% distrust rate
for a routine HDFC OTP is implausible on its face, which is the tell.

**What changes, and what does not.**
- The H1 analysis is unaffected and stands as pre-registered. On scam stimuli the
  resistant coding is valid and both arms were properly differentiated.
- H2's numbers are no longer reported as a finding. `analyze_study.py` still
  computes them — deleting them would hide the error — but prints them under an
  explicit UNINTERPRETABLE header with both defects and the ignore-share
  diagnostic, so the number cannot travel without its caveat.
- The sections above are **not** edited. This document's purpose is to record what
  was committed to in advance, including where that commitment turned out to be
  flawed.

**Corrected instrument (built, not run).** The measure H2 needed is a direct
judgment of the message, not an inferred one: *"Do you think this message is
genuine? (Y/N)"*, asked per stimulus alongside the existing action question. On a
legitimate stimulus, answering "no" is a false alarm regardless of what action the
participant would take, which separates the two things the current coding fuses.
This has been added to `run_study.py`, the spreadsheet export/import, and
`analyze_study.py`, and is reported for any future run. It is deliberately left
unrun: re-running with new participants is out of scope, and back-filling the
existing 20 participants is not possible without re-contacting them, which the
data-handling section forbids by design (ids are random and unlinked to identity).

No other section of this pre-registration is amended.
