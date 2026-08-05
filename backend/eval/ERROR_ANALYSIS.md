# Error analysis + class-imbalance story

Written against the real gold set (N=250, `data/corpus/gold/labels_primary.jsonl`,
every message individually reviewed, single annotator — no Cohen's kappa yet).
Baseline/teacher/student numbers below are from the 2026-08-05 run of
`eval/compare.py`, after fixing a train/eval leakage bug in `eval/distill.py`
(the student had been trained on 190 of the 250 gold messages; retrained
clean — see `eval/distill.py`'s `load_training_data` docstring).

## Headline numbers

| Model | macro-F1 | macro-P | macro-R | latency/msg | size |
|---|---|---|---|---|---|
| Baseline (TF-IDF + LR) | 0.49 | 0.44 | 0.62 | ~0ms | negligible |
| Teacher (qwen2.5:7b, Ollama) | 0.52 | 0.82 | 0.43 | 8120ms | ~4.5GB |
| Student (distilled ModernBERT) | 0.56 | 0.51 | 0.74 | 65ms | 574MB / 150M params |

**Read this cautiously, not triumphantly.** Two methodology caveats apply
equally to all three rows above, so the *ranking* is fair, but the absolute
numbers are optimistic:

1. **Per-technique thresholds are tuned by sweeping a grid against this same
   250-message set** (`eval/metrics.py`, `THRESHOLD_GRID`), then P/R/F1 are
   reported at the winning threshold. That's in-sample threshold selection,
   not held-out threshold calibration — every number here is an upper bound
   on what a truly fresh threshold would give, especially for techniques
   with only a handful of positives (below).
2. **7 of 11 techniques have single-digit-to-low-double-digit support** in
   the 250-message gold set (`sunk_cost_pressure`=3, `trust_transfer`=4,
   `payment_irreversibility`=4, `fake_scarcity`=5, `verification_theater`=8,
   `isolation`=11, `channel_switch`=12). Macro-averaging weights these
   exactly as heavily as `manufactured_urgency` (36) or `false_authority`
   (33), so a single misprediction on a 3-support class swings its F1 by
   0.33+ and drags the macro average with it. Read per-technique rows, not
   just the macro line.

The plan's stated risk was "the student may lose badly to the teacher on
rare techniques with few positives." That didn't happen here — the student
is competitive or ahead on every technique except `channel_switch` — but
given caveat #2, treat the 0.52-vs-0.56 gap between teacher and student as
"roughly tied, with a large latency/size win for the student," not as a
confidently-established win.

## Concrete failure: the teacher is blind to `trust_transfer`

Teacher's `trust_transfer` row: P=0.00, R=0.00, F1=0.00 (N=4) — total
failure, not a threshold artifact. Live-reran `classify()` on all 4 gold
`trust_transfer` positives to confirm:

| id | message (truncated) | true labels | teacher output |
|---|---|---|---|
| IN-016 | "Mom I lost my phone, this is my friend's number. Can you quickly send Rs 15000..." | trust_transfer | `reciprocity_hook` (0.9) |
| IN-038 | "Dad don't tell mom I called, but I'm stuck and need Rs 20,000 transferred..." | trust_transfer, isolation | `isolation` (1.0) only |
| IN-017 | "Hi it's me, changed my number after the SIM issue... don't call the old number" | trust_transfer, manufactured_urgency, channel_switch | *(nothing)* |
| uci-03864 | "Oh my god! I've found your number again! I'm so glad, text me back..." | trust_transfer | *(nothing)* |

IN-016 is the textbook "Hi Mum" scam — about as clean an example of
impersonating a specific trusted person as this taxonomy has — and the
7B teacher not only missed it, it actively mislabeled it as
`reciprocity_hook`. The model appears to have no real notion of "this
message is pretending to be someone the reader already knows," as
distinct from `false_authority` (impersonating an institution) or
`isolation` (the secrecy instruction that often rides alongside it, and
which the teacher does pick up in IN-038). The student does somewhat
better on this technique (F1=0.46) but is far from solved either — see below.

## Concrete failure: the student barely learned `channel_switch`

Student `channel_switch`: F1=0.24, R=0.17 (N=12) — the weakest technique
for the student and the one place the plan's predicted "student loses to
teacher on rare techniques" risk actually shows up (teacher R=0.17 too, but
teacher P=1.00 vs student P=0.40 — both models are bad here, in different
ways). Real misses, with the student's raw probability against a 0.20
decision threshold:

- `uci-02941` ("You have 1 new message. Please call 08712400200.") — proba 0.00
- `IN-012` ("...we cannot continue on a recorded government line. Please call me back on this personal WhatsApp number instead.") — proba 0.06
- `IN-017` (same "changed my number" message as above) — proba 0.03

Only 13 of the 828 student training examples are positive for
`channel_switch` (see the `load_training_data` support printout). That's
almost certainly why: there isn't enough signal in training for the model
to generalize the pattern ("move off this channel") beyond a few
surface phrases, so it defaults to near-zero confidence on held-out
paraphrases of the same idea.

## Concrete failure: `payment_irreversibility` over-fires on generic "Rs + urgency" text

Student `payment_irreversibility`: P=0.14, R=0.75, F1=0.24 (N=4) — the
opposite failure mode from `channel_switch`. Instead of under-firing, the
model appears to have learned "mentions a Rupee amount + urgency/fee
language" as a loose proxy for the technique, rather than its actual
definition (a payment *framed as a receipt* — a UPI collect request or
"scan to receive" trick). False positives include messages whose real
technique is `sunk_cost_pressure`, `fake_scarcity`, or `false_authority`
(different manipulation entirely, just also money-shaped):

- `IN-035` — "You've already paid the Rs 12,000 registration fee... pay the final Rs 8,000..." (true: `sunk_cost_pressure`, not `payment_irreversibility`) — proba 0.34
- `IN-020` — "Only 3 slots left... Deposit Rs 10,000 now..." (true: `fake_scarcity`) — proba 0.20
- `IN-041` — "Please switch to Google Meet..." (true: `channel_switch`) — proba 0.22

More concerning for a product that must never render "safe": it also
false-positives on **entirely benign transactional messages** —
`IN-B01` (a legitimate Amazon OTP notice, proba 0.17), `IN-B06` (an
electricity bill reminder, proba 0.15), `IN-B15` (a Myntra refund
confirmation, proba 0.20), and `IN-B10` (a bank's own PSA telling users
never to share their PIN/OTP — proba 0.28, and it also nearly trips
`channel_switch` at 0.28). The model is picking up "Rs amount + payment
verb" as a surface cue shared by real scam messages, legitimate bank
notices, *and* anti-scam advisories alike. The `IN-B10` case is the most
interesting of these: a message *warning people about scams* gets flagged
because it uses the same vocabulary scams use. With only 11 positive
training examples for this technique, the model doesn't have enough
contrast between "payment framed as receipt" and "any message involving a
payment" to draw a tighter boundary.

Worth noting: this same conflation shows up in the teacher too. Testing
`IN-043` ("you've already paid Rs 60,000... final Rs 25,000 embassy fee or
lose it") live — true labels are `manufactured_urgency`, `false_authority`,
`fear_of_consequence`, `sunk_cost_pressure` — the teacher predicted
`payment_irreversibility` (0.9) and nothing else, missing all four true
labels and reaching for the same "money mention → payment_irreversibility"
shortcut. This isn't just a student-distillation artifact; it looks like
the taxonomy itself is hard to disambiguate from money-adjacent language
alone, for both model sizes.

## Concrete failure: `trust_transfer` vs `isolation`/`false_authority` confusion

The gold-labeling legend (`eval/export_labeling_sheet.py`'s Legend sheet)
explicitly calls out `false_authority` (impersonates an institution) vs
`trust_transfer` (impersonates a specific known person) as a common
human mix-up. The student reproduces the same confusion mechanically: its
`trust_transfer` false positives are almost all `isolation` +
`false_authority` messages (police/officer-impersonation "stay on the
call, don't tell your family" scripts) — `IN-004`, `IN-003`, `IN-034`,
`IN-028`, `IN-008` — none of which involve impersonating a specific known
person. The model seems to have partly learned "secrecy instruction +
authority claim" as a package deal that includes `trust_transfer`, when
in this taxonomy the two are meant to be distinguished by *who* is being
impersonated (an institution vs. someone specific to the reader).

## What this means

- **The distillation thesis holds up under a clean, leakage-free eval**:
  the 574MB / 65ms student is not just "acceptable" next to the 4.5GB /
  8120ms teacher — it's at least competitive on accuracy and clearly
  better on the metrics that matter for a live product. That's the
  strongest available result for a resume artifact making the case for
  distillation.
- **Neither model is close to reliable on low-support techniques** —
  `channel_switch`, `payment_irreversibility`, `sunk_cost_pressure`,
  `trust_transfer`, `fake_scarcity` all have single-digit-to-low-teens
  support in gold and correspondingly shaky, sometimes contradictory,
  per-technique behavior. This is a corpus-size problem, not obviously a
  model-choice problem: both a 7B LLM with a hand-written prompt and a
  from-scratch fine-tune fail the same technique (`trust_transfer`,
  `payment_irreversibility`) in overlapping ways.
- **False positives on benign transactional/advisory messages
  (`IN-B01`, `IN-B06`, `IN-B10`, `IN-B15`) are the finding to take most
  seriously** for a "never say safe" product — the failure mode isn't
  hypothetical, it's observed: an anti-scam PSA gets partially flagged as
  itself manipulative.
- **Macro-F1 alone is a misleading single number here** given how much of
  it is driven by 3-12-support classes; any report of this eval should
  lead with the per-technique table, not the macro line in isolation.

## Layer 2: validator pass rate (found broken, then fixed)

This is Eval A (the classifier). The plan's stated bigger technical risk
was Layer 2 — the LLM-written purpose sentence that quotes the manipulated
span verbatim and explains why it's being used. Risk #2 in the plan says
explicitly: *"if the validator pass-rate is poor, report the number."*

**It was poor — 0%.** Checking `study/stimuli_cache.json` (the frozen
treatment-arm text for the human study) found all 7 explanation cards were
table-only fallback text, not LLM-generated. Root cause: the old schema
asked the model to do two things in one free-text field — quote the span
verbatim *and* explain its purpose. Live re-generation showed the model
almost never did the first part (6/7 cases: "missing verbatim span"; the
model paraphrased the message instead of copying it, even with an
explicit instruction and a 40-word budget). The prompt's own example
didn't demonstrate a quote either, which likely reinforced the pattern.

**Fix:** narrowed the model's job to the purpose clause only, and insert
the verbatim quote programmatically from the Layer-1 span — the same
"guarantee by construction, not by hoping a validator catches a drift"
approach already used for the reality-check/counter-action text (see
`app/explain.py`'s module docstring). Re-verified live against all 60
`handcrafted_india` messages after the fix: **59/59 cards (100%) passed
validation on the first attempt, 0 retries, 0 fallbacks** (the 60th
message hit the pre-existing Layer-1 Pydantic confidence-validation bug
during `classify()`, same as the 4/250 errors noted above — no detections
were available to test for that one). `study/stimuli_cache.json` has been
rebuilt with real generated explanations for all 4 scam stimuli.

This is the more consequential finding of the two evals: the classifier
being mid (macro-F1 0.49-0.56) is a defensible, pre-registered-as-expected
outcome given corpus size. Layer 2 silently degrading to boilerplate on
100% of the study's treatment-arm content would have invalidated the
human study's entire premise, since Eval B is testing whether the named,
purpose-explained technique changes behavior versus a bare red-banner
flag — and a fallback sentence like `"X" is a sign of Y.` is not
meaningfully different from a flag with a label attached.

## Not covered here

- Inter-annotator agreement (Cohen's kappa) — no second annotator yet, so
  it's still unknown how much of the "true" label for a borderline message
  (e.g. is `IN-B10` really zero-technique, or does its PIN/OTP language
  deserve a low-confidence tag?) reflects real signal vs. one annotator's
  judgment call.
- A held-out threshold-calibration split — the in-sample tuning caveat
  above means a true generalization number for all three models is still
  unmeasured.
