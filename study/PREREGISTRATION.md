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
identity), arm, stimulus id, response code, confidence, timestamp. See CLAUDE.md —
no PII is collected as part of this study; participants are not asked to submit
their own real messages.

---

## Amendments

*(none yet — any change made after the first participant is logged here with a
date and reason, not edited into the sections above)*
