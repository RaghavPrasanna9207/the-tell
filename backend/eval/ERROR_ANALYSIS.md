# Error analysis + class-imbalance story

Written against the real gold set (N=250, `data/corpus/gold/labels_primary.jsonl`,
every message individually reviewed by a single primary annotator; see the
2026-08-23 update below for Cohen's kappa against a second annotator on an
80-message subset).

## 2026-08-23: Cohen's kappa on the 80-message subset (Phase A4)

A second annotator labeled the same 80-message kappa subset via the Excel
handoff (`eval/export_labeling_sheet.py --annotator second`, filled in
independently with no access to the primary labels, then
`eval/import_labeling_sheet.py --annotator second`). `eval/kappa.py` compares
`labels_primary.jsonl` vs `labels_second.jsonl` on the shared 80 ids.

**Macro-average kappa: 0.68** (over all 11 techniques) — above the plan's 0.6
threshold, so per the plan's own rule ("if kappa < 0.6, fix the taxonomy, not
the number") the taxonomy is not flagged as broken overall. Message-level
agreement, which is better-powered because it uses all 80 messages rather
than a per-technique sliver: "any technique present" binary kappa 0.66, exact
label-set match rate 0.70, mean Jaccard similarity 0.79.

Per-technique, most techniques land in the 0.66-1.00 range with reasonable
agreement. One technique is a genuine outlier: **`reciprocity_hook` kappa =
-0.02** (primary marked it 14 times, second only once — essentially no
correlation, not just low support). Looking at the disagreements, this isn't
noise, it's a real definitional split: the primary annotator marked classic
UK-lottery-style spam ("you have won a £2000 prize, call to claim", "free
ringtone waiting for you") as `reciprocity_hook` (the "free gift" *is* the
reciprocity hook), while the second annotator treated these as `none` or
`manufactured_urgency`/`channel_switch` only, apparently reading
`reciprocity_hook` more narrowly as requiring an explicit unsolicited-favor
setup (closer to the "Mom I lost my phone" / "I sent you money by mistake"
pattern) rather than a bare prize-claim hook. 7 of the 24 total label-set
disagreements are exactly this pattern (`uci-00042`, `uci-00056`,
`uci-00095`, `uci-01970`, `uci-02119`, `uci-02525`, `uci-02987`, `uci-03010`,
`uci-03409`, `uci-04086`, `uci-04754`, `uci-04841` — the `reciprocity_hook`
column specifically).

**Fixed the same day**, since `counter_moves.yaml`'s own source for this
technique ("I4C advisory on lottery and prize-based frauds") already backed
the primary annotator's broader reading — the taxonomy's intent was never
ambiguous, only its written wording was. `app/taxonomy.py`'s
`RECIPROCITY_HOOK` enum docstring and `TECHNIQUE_DESCRIPTIONS` entry (used
both in the Layer 1 classifier prompt and the labeling legend, per the
"keep the two in sync" note in that file) now say explicitly that a bare
prize-claim counts on its own, with no separate "do me a favor in return"
ask required. `eval/export_labeling_sheet.py`'s Legend sheet got a matching
"Common mix-ups" line. `gold_labeling_second.xlsx` was regenerated from the
fixed definitions (labels unchanged — the 80 existing answers round-tripped
back in via `labels_second.jsonl`).

The recorded 0.68 macro-kappa / -0.02 `reciprocity_hook` kappa above is the
honest, un-fudged number from *before* this fix and is kept as-is — it's the
evidence that motivated the fix, not something to retroactively improve by
relabeling. A future kappa run against a definition-aware second annotator
would be the real test of whether the fix worked, not a re-score of the same
80 answers.

A smaller secondary pattern: the second annotator layered `verification_theater`
on top of the primary's labels on several digital-arrest-style messages
(`IN-003`, `IN-004`, `IN-006`, `IN-011`) that mention an FIR copy, an ID card,
or "funds verified" — reading those mentions as verification theater more
liberally than the primary did. Worth a note in the legend, but with
`verification_theater` kappa = 0.71 on n=4/7 support this is closer to normal
borderline disagreement than the `reciprocity_hook` split.

Several other techniques (`isolation`, `verification_theater`,
`channel_switch`, `payment_irreversibility`, `trust_transfer`, `fake_scarcity`,
`sunk_cost_pressure`) have single-digit support even on this subset — their
kappa values are individually close to statistical noise (see `eval/kappa.py`'s
support-flagging) and shouldn't be read as precise per-technique reliability
scores, only as "not contradicted by this sample."

## 2026-08-18: the student is now a real part of the running system

Previously the trained ModernBERT student existed only as a benchmark
artifact — `/analyze` called the teacher directly, on every message, at
~5s/message. That's now fixed: `app/student.py` runs the student first as a
deliberately high-recall **gate**, and the teacher is only woken up when the
student finds something worth investigating (see `app/main.py`). The student
has no notion of verbatim spans, so it can never answer on its own — Layer
2's span-grounding guarantee is untouched, since a "flagged" message still
goes through the full teacher + explain pipeline exactly as before.

`eval/gate_calibration.py` measures the trade-off against the real gold set
at `GATE_THRESHOLD = 0.05`:

| threshold | scam recall | clean wake rate | overall wake rate |
|---|---|---|---|
| 0.05 (chosen) | 0.87 | 0.07 | 0.29 |

Read: at this threshold, 87% of scam messages correctly wake the teacher (a
miss here means Layer 2 never runs for that message — the real cost of the
gate being wrong), only 7% of clean messages needlessly wake the teacher
(pure latency cost, not a correctness cost — the teacher would presumably
also find nothing), and only 29% of all gold messages reach the teacher at
all — meaning roughly 71% of traffic resolves on the fast, student-only path.
The gate's own 0.87 recall is lower than the teacher's held-out 0.43
per-technique macro-recall would suggest is achievable, because "does
*anything* fire" (message-level) is an easier bar than "does the *correct*
technique fire" (per-technique) — the gate only needs to decide whether the
teacher is worth calling, not get the technique right itself.

**A real deployment gotcha this surfaced:** the first call to the student
model in a fresh process — first-time `torch`/`transformers` import, weight
load, GPU transfer — measured ~20-30s. Loading it lazily on the first
request would have meant whichever real user happened to send the first
message after a server restart eating that delay. Fixed with a FastAPI
`lifespan` startup hook (`app/main.py`) that calls `app.student.warm_up()`
before the server accepts any request, so the cost is paid once at boot.
Confirmed live: after warm-up, back-to-back requests settle to 125-203ms —
consistent with the 65-175ms range `eval/gate_calibration.py` measures
offline, and nowhere close to the 20-30s cold figure.

## 2026-08-18 update: held-out threshold calibration changes the ranking

The numbers below were originally reported **in-sample**: per-technique
thresholds were tuned by sweeping a grid against the same 250-message set
they were then scored on (`eval/metrics.py`'s old `per_technique_metrics`),
flagged at the time as caveat #1 below — "every number here is an upper
bound." `eval/metrics.py` now also has `per_technique_metrics_cv`: 5-fold
threshold calibration where each fold's threshold is chosen on the *other*
folds only, so every message is scored by a threshold that never saw its
label. `eval/compare.py` and `eval/run_eval.py` now print both, and the
**held-out numbers are the ones that matter** — the in-sample numbers are
kept below only as the visible measure of how much the old methodology was
overstating things.

**The ranking changes, not just the absolute numbers.** In-sample, the
distilled student looked like it matched or beat the teacher (0.52 vs 0.56).
Held-out, that reverses: the teacher is back in front.

| Model | macro-F1 (in-sample) | macro-F1 (held-out) | latency/msg | size |
|---|---|---|---|---|
| Baseline (TF-IDF + LR) | 0.49 | **0.41** | ~0ms | negligible |
| Teacher (qwen2.5:7b, Ollama) | 0.53 | **0.51** | 4966ms | ~4.5GB |
| Student (distilled ModernBERT) | 0.56 | **0.48** | ~65-175ms | 574MB / 150M params |

(In-sample teacher macro-F1 above is 0.53, not the 0.52 originally reported —
this run used a live teacher pass, and the 7B model isn't perfectly
deterministic call to call even at low temperature; a ~0.01 wobble on
re-running is expected, not a regression.)

**Latency correction (2026-08-18, same day, caught before it shipped
anywhere else):** an earlier pass through this same 250-message set measured
student latency at 14ms/msg via `eval/compare.py`'s `student_predict`, run
immediately after ~20 minutes of continuous teacher inference on the same
GPU. That number was a measurement artifact — the student inherited boosted
GPU clocks it doesn't sustain on its own, not a real steady-state figure.
Repeated clean measurements (fresh process, no preceding GPU load) of the
same 250 messages land in the **65-175ms/msg** range; `nvidia-smi` confirms
this laptop's RTX 4060 idles at 1890MHz against a 3105MHz boost ceiling and
doesn't hold a boost clock for one-message-at-a-time inference, which is
exactly how a live product actually calls it. Table above and all latency
claims below use the corrected range.

**Read this as: the student is a genuinely good deal, not a free lunch.**
Distillation buys a real 28-76x latency reduction (65-175ms vs the teacher's
4966ms, not the erroneous 355x an earlier measurement implied) and ~8x
smaller footprint (150M vs ~7B params) for a real but modest 0.03 macro-F1
cost versus the teacher — and the student still clearly beats the TF-IDF
baseline. That's a different, more honest headline than "student ties or
beats the teacher," and it survived the stricter methodology rather than
being *revealed* as false by it, since the teacher's own in-sample-to-held-out
drop (0.53→0.51) was smaller than the student's (0.56→0.48) — the student's
original number was benefiting more from in-sample overfitting on its
low-support techniques (see the `channel_switch` and
`payment_irreversibility` sections below, both revised with the held-out
numbers).

The plan's stated risk was "the student may lose badly to the teacher on
rare techniques with few positives." **That risk is now visible where it
wasn't before**: the student's held-out `payment_irreversibility` collapses
to F1=0.00 (was a misleadingly decent-looking 0.24 in-sample), and
`channel_switch` drops from 0.24 to 0.12. Both are discussed below. The two
methodology caveats from the original write-up still apply and now compound
with the held-out result:

1. ~~Per-technique thresholds are tuned in-sample~~ — **fixed above**, but
   worth remembering *why* it mattered this much: with 250 messages and
   5-fold CV, each fold's threshold is chosen from ~200 examples, and for a
   3-11-support technique that's only 2-9 positives to tune against per
   fold. The held-out numbers are honest, not stable — rerunning with a
   different seed would move the low-support rows more than the
   well-supported ones.
2. **7 of 11 techniques have single-digit-to-low-double-digit support** in
   the 250-message gold set (`sunk_cost_pressure`=3, `trust_transfer`=4,
   `payment_irreversibility`=4, `fake_scarcity`=5, `verification_theater`=8,
   `isolation`=11, `channel_switch`=12). Macro-averaging weights these
   exactly as heavily as `manufactured_urgency` (36) or `false_authority`
   (33), so a single misprediction on a 3-support class swings its F1 by
   0.33+ and drags the macro average with it. Read per-technique rows, not
   just the macro line.

## Original headline numbers (in-sample, kept for comparison)

Baseline/teacher/student numbers below are from the 2026-08-05 run of
`eval/compare.py`, after fixing a train/eval leakage bug in `eval/distill.py`
(the student had been trained on 190 of the 250 gold messages; retrained
clean — see `eval/distill.py`'s `load_training_data` docstring).

| Model | macro-F1 | macro-P | macro-R | latency/msg | size |
|---|---|---|---|---|---|
| Baseline (TF-IDF + LR) | 0.49 | 0.44 | 0.62 | ~0ms | negligible |
| Teacher (qwen2.5:7b, Ollama) | 0.52 | 0.82 | 0.43 | 8120ms | ~4.5GB |
| Student (distilled ModernBERT) | 0.56 | 0.51 | 0.74 | 65ms | 574MB / 150M params |

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

Student `channel_switch` held-out: F1=0.12, P=0.25, R=0.08 (N=12) — the
weakest technique for the student and the one place the plan's predicted
"student loses to teacher on rare techniques" risk actually shows up (teacher
held-out F1=0.38 — clearly ahead here). The in-sample number (F1=0.24) was
already the worst student row before the held-out fix; held-out is worse
still, because the mean decision threshold (0.21) hides real fold-to-fold
threshold instability — with only 13 positive training examples spread
across 5 folds, whichever fold happens to get an unlucky subset of those 13
for its threshold-selection step ends up with a threshold that misses most
of that fold's true positives. Real misses, with the student's raw
probability (unaffected by which threshold is chosen — only the pass/fail
cutoff changed under CV):

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

Student `payment_irreversibility` held-out: P=0.00, R=0.00, F1=0.00 (N=4) —
a complete collapse from the in-sample P=0.14/R=0.75/F1=0.24 originally
reported here, and the single biggest number in this whole analysis to move
under the methodology fix. This is not just "the threshold moved" — cost=47
with 0 true positives means the held-out predictions include real false
positives (the same over-firing behavior described below) while catching
*none* of the 4 true positives, across every fold. That's a signal the raw
probabilities themselves don't reliably separate the true positives from the
noise, not merely that the in-sample threshold happened to be tuned to catch
them by luck. Instead of under-firing (`channel_switch`'s failure mode), the
model appears to have learned "mentions a Rupee amount + urgency/fee
language" as a loose proxy for the technique, rather than its actual
definition (a payment *framed as a receipt* — a UPI collect request or
"scan to receive" trick). False positives include messages whose real
technique is `sunk_cost_pressure`, `fake_scarcity`, or `false_authority`
(different manipulation entirely, just also money-shaped) — probabilities
below are raw model output, unaffected by which threshold is applied:

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

- **The distillation thesis holds up under a clean, leakage-free, held-out
  eval — as a latency/size trade, not a free accuracy win.** The 574MB /
  ~65-175ms student trails the 4.5GB / 4966ms teacher by 0.03 macro-F1 (0.48
  vs 0.51) once thresholds are calibrated honestly, reversing the earlier
  in-sample result that had the student ahead. That's a materially
  different but still strong resume artifact story: 28-76x faster (see the
  latency correction above — an earlier "355x" figure was a measurement
  artifact, not real) and ~8x smaller for a small, real accuracy cost — and
  that cost has a clean
  support-driven split, not a uniform quality gap across the taxonomy. On
  the 4 highest-support techniques (`manufactured_urgency`=36,
  `false_authority`=33, `reciprocity_hook`=28, `fear_of_consequence`=24) the
  student clearly beats the teacher (e.g. `fear_of_consequence` 0.88 vs
  0.55). On the 7 lower-support techniques (support 3-12) the teacher wins
  5 of 7, loses only `trust_transfer` (where it's completely blind, see
  above) and roughly ties `isolation`. The honest summary: **the student is
  the better choice on the techniques with enough training data, and the
  teacher's edge is concentrated exactly where the student had too few
  positive examples to generalize from** — a distillation story about data,
  not about the smaller architecture being categorically worse.
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
validation on the first attempt, 0 retries, 0 fallbacks** (the 60th message
hit the Layer-1 out-of-range-confidence bug during `classify()` — no
detections were available to test for that one; that bug is now fixed, see
next paragraph). `study/stimuli_cache.json` has been rebuilt with real
generated explanations for all 4 scam stimuli.

**2026-08-18: the out-of-range-confidence bug above is now fixed.** The
teacher occasionally emits a `confidence` outside [0, 1] — schema-constrained
decoding enforces JSON shape, not declared numeric range — which used to
raise an uncaught `pydantic.ValidationError`. In `eval/compare.py`'s live
teacher pass, this showed up as gold messages "erroring" and being counted
as all-zero predictions, quietly undercounting teacher recall; in the
FastAPI app it would have been an unhandled 500 on `/analyze`. `app/llm.py`'s
`generate_structured` now clamps a pure range violation to its declared
bound and logs it, rather than raising. Confirmed live in the same run that
produced the held-out numbers above: the teacher pass over all 250 gold
messages clamped 2 out-of-range confidences and had **0 messages error out**
— previously this class of bug was silently costing a few gold messages'
worth of teacher recall on every eval run.

This is the more consequential finding of the two evals: the classifier
being mid (held-out macro-F1 0.41-0.51 across baseline/student/teacher) is
a defensible, pre-registered-as-expected outcome given corpus size. Layer 2
silently degrading to boilerplate on
100% of the study's treatment-arm content would have invalidated the
human study's entire premise, since Eval B is testing whether the named,
purpose-explained technique changes behavior versus a bare red-banner
flag — and a fallback sentence like `"X" is a sign of Y.` is not
meaningfully different from a flag with a label attached.

## Not covered here

- ~~Inter-annotator agreement (Cohen's kappa)~~ — done, see the 2026-08-23
  update at the top of this file. Still open: it's only measured on an
  80-message subset, and only against `labels_primary.jsonl`'s judgment calls
  — e.g. whether `IN-B10` is really zero-technique isn't itself re-litigated,
  only how consistently a second person would apply the same taxonomy.
  ~~A held-out threshold-calibration split~~ — done, see the 2026-08-18
  update at the top of this file.
