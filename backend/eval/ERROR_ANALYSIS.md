# Error analysis + class-imbalance story

Written against the gold set (N=250, `data/corpus/gold/labels_primary.jsonl`),
with every message individually reviewed by a single primary annotator. The
2026-08-23 update below has Cohen's kappa against a second annotator on an
80-message subset.

## 2026-09-08: what the cascade gate costs, end to end

Every number in this file below this section scores a *component*. The product
is the cascade: `app/main.py` runs the student gate first, and a message the
gate rejects returns "no manipulation techniques detected" without the teacher
ever being called. That composition had never been measured. `compare.py`
scores the teacher as if it ran on everything, and `gate_calibration.py` scores
the gate alone at the message level ("does anything fire").
`eval/cascade_eval.py` closes the gap.

Method: one teacher pass over all 250 gold messages, scored twice, once whole
and once with gate-rejected rows zeroed out. The teacher isn't deterministic
call to call, so two separate passes would confound the gate's cost with
sampling noise. Scoring one pass twice means the only difference between the
two rows is the gate.

| | macro-P | macro-R | macro-F1 |
|---|---|---|---|
| Teacher alone | 0.81 | 0.47 | 0.56 |
| Deployed cascade | 0.81 | 0.46 | 0.55 |

The gate costs 0.01 macro-F1. Only `reciprocity_hook` moved (0.55 to 0.51);
every other technique is identical. For a component that removes 71% of teacher
calls, that is a good trade, and a much better one than the "student loses to
teacher on rare techniques" risk predicted.

For a "never say safe" product, the correctness cost is the number to watch:
9 of 69 scam messages are silenced by the gate, and Layer 2 never runs for
them. All nine:

`uci-03010`, `uci-04841`, `uci-04754`, `uci-04086`, `uci-02119`, `uci-00056`,
`uci-00833`, `uci-03864`, `uci-02941`

Every one is a UCI message; not a single `IN-*` message is silenced. The gate
fails on 2000s UK SMS spam (ringtone offers, prize-claim lines, "call 09xxx")
and holds on the India-specific digital-arrest and UPI scripts the product is
for. That is the favorable reading of this result, but it is favorable partly
by construction: the student's India-specific training data and the `IN-*`
eval messages share an author, so in-domain performance here is measured on
the distribution the model was taught. The UCI misses are the ones that look
like out-of-distribution failure.

Most of them turn out not to be clean failures at all. Six of the nine fall
inside the 80-message kappa subset, so a second annotator independently
labeled them. On all six the second annotator disagreed with the primary, and
on five of six they said the message contained *no technique*:

| id | primary | second annotator |
|---|---|---|
| `uci-03010` | `channel_switch` | *(none)* |
| `uci-04754` | `reciprocity_hook` | *(none)* |
| `uci-04086` | `reciprocity_hook` | *(none)* |
| `uci-02119` | `manufactured_urgency`, `reciprocity_hook` | *(none)* |
| `uci-00056` | `manufactured_urgency`, `reciprocity_hook` | *(none)* |
| `uci-04841` | `manufactured_urgency`, `channel_switch`, `reciprocity_hook` | `manufactured_urgency` |

Every one is the `reciprocity_hook` / prize-claim definitional split
documented in the 2026-08-23 kappa section below, the single technique where
agreement collapsed to kappa = -0.02.

So the gate's headline correctness cost is measured against labels a second
human would largely not have assigned. On these messages the gate is siding
with annotator 2, and "9 of 69 silenced" is an upper bound on the gate's error
rather than an estimate of it. The true figure depends on a taxonomy question
the two annotators answered differently. This is the clearest demonstration in
the project of why the kappa exercise was worth doing: without it, these nine
would read as nine clean model failures.

Caveat on the absolute numbers: this run's teacher scored macro-F1 0.56, where
the table further down this file records 0.51 from an earlier pass. A clean
re-run on 2026-09-09 also returned 0.56, so 0.51 is the outlier, and the README
figure has been corrected. Baseline (0.41) and student (0.48) reproduced
exactly across all runs. The teacher's instability is consistent with the
fold-to-fold behavior documented below: `sunk_cost_pressure` (support 3) scored
a perfect 1.00 in one run, and one flip on a 3-support class moves the macro
average ~0.03.

The cascade delta of 0.01 holds regardless, because both rows come from the
same teacher pass. That is the reason for scoring one pass twice instead of
running two separate passes.

## 2026-09-09: latency on this hardware is not a publishable number

Three runs of the same 250-message pass, same script path, same model:

| Run | Teacher | Student / gate |
|---|---|---|
| 2026-08-18 | 4966 ms/msg | 65-175 ms/msg |
| 2026-09-08 (`cascade_eval.py`, gate ran first) | 523 ms/msg | 315 ms/msg |
| 2026-09-09 (`compare.py`, clean) | 2794 ms/msg | 216 ms/msg |

The teacher spans 9.5x. Whether Ollama already had the 4.5GB model resident,
and what clock state the GPU was in, plausibly dominate the signal. The
student's *fastest* number (216ms) comes from the run where it executed right
after a 20-minute teacher pass, the same condition behind the discredited 14ms
artifact documented below. Its slowest (315ms) comes from the run where it went
first on an idle GPU. The bias has a consistent direction, which makes it a
systematic measurement problem rather than noise.

All latency point estimates have therefore been removed from the README's
results table. Publishing the flattering end of a 9.5x range is the error this
file already documents once. What survives, and doesn't depend on wall-clock
time: the student is ~30x smaller in parameters, and the gate eliminates 71% of
teacher calls.

A latency claim would need fixed clocks (`nvidia-smi -lgc`), a warmed resident
model, and several interleaved runs. That work should happen before any latency
number goes back in.

## 2026-08-23: Cohen's kappa on the 80-message subset (Phase A4)

A second annotator labeled the same 80-message kappa subset via the Excel
handoff (`eval/export_labeling_sheet.py --annotator second`, filled in
independently with no access to the primary labels, then
`eval/import_labeling_sheet.py --annotator second`). `eval/kappa.py` compares
`labels_primary.jsonl` and `labels_second.jsonl` on the shared 80 ids.

Macro-average kappa is 0.68 over all 11 techniques. That is above the plan's
0.6 threshold, so under the plan's own rule ("if kappa < 0.6, fix the
taxonomy, not the number") the taxonomy is not flagged as broken overall.
Message-level agreement is better-powered because it uses all 80 messages
instead of a per-technique sliver: "any technique present" binary kappa 0.66,
exact label-set match rate 0.70, mean Jaccard similarity 0.79.

Per-technique, most techniques land in the 0.66-1.00 range with reasonable
agreement. One technique is an outlier: `reciprocity_hook` kappa = -0.02. The
primary annotator marked it 14 times and the second only once, which is
essentially no correlation, more than low support alone explains. The
disagreements show a definitional split. The primary annotator marked classic
UK-lottery-style spam ("you have won a £2000 prize, call to claim", "free
ringtone waiting for you") as `reciprocity_hook`, treating the "free gift" as
the reciprocity hook. The second annotator treated these as `none` or as
`manufactured_urgency`/`channel_switch` only, apparently reading
`reciprocity_hook` more narrowly as requiring an explicit unsolicited-favor
setup (closer to the "Mom I lost my phone" / "I sent you money by mistake"
pattern) instead of a bare prize-claim hook. 7 of the 24 total label-set
disagreements are exactly this pattern (`uci-00042`, `uci-00056`, `uci-00095`,
`uci-01970`, `uci-02119`, `uci-02525`, `uci-02987`, `uci-03010`, `uci-03409`,
`uci-04086`, `uci-04754`, `uci-04841`, counting the `reciprocity_hook` column
specifically).

This was fixed the same day. `counter_moves.yaml`'s own source for this
technique ("I4C advisory on lottery and prize-based frauds") already backed the
primary annotator's broader reading, so the taxonomy's intent was clear and
only its written wording was ambiguous. `app/taxonomy.py`'s `RECIPROCITY_HOOK`
enum docstring and `TECHNIQUE_DESCRIPTIONS` entry (used in both the Layer 1
classifier prompt and the labeling legend, per the "keep the two in sync" note
in that file) now say explicitly that a bare prize-claim counts on its own,
with no separate "do me a favor in return" ask required.
`eval/export_labeling_sheet.py`'s Legend sheet got a matching "Common mix-ups"
line. `gold_labeling_second.xlsx` was regenerated from the fixed definitions
with labels unchanged: the 80 existing answers round-tripped back in via
`labels_second.jsonl`.

The recorded 0.68 macro-kappa and -0.02 `reciprocity_hook` kappa above are the
un-fudged numbers from *before* this fix and are kept as-is. They are the
evidence that motivated the fix, and relabeling to improve them retroactively
would erase that evidence. The test of whether the fix worked is a future kappa
run against a definition-aware second annotator; re-scoring the same 80
answers would not test it.

A smaller secondary pattern: the second annotator layered
`verification_theater` on top of the primary's labels on several
digital-arrest-style messages (`IN-003`, `IN-004`, `IN-006`, `IN-011`) that
mention an FIR copy, an ID card, or "funds verified", reading those mentions as
verification theater more liberally than the primary did. It's worth a note in
the legend, but with `verification_theater` kappa = 0.71 on n=4/7 support this
is closer to normal borderline disagreement than the `reciprocity_hook` split.

Several other techniques (`isolation`, `verification_theater`,
`channel_switch`, `payment_irreversibility`, `trust_transfer`, `fake_scarcity`,
`sunk_cost_pressure`) have single-digit support even on this subset. Their
kappa values are individually close to statistical noise (see `eval/kappa.py`'s
support-flagging) and should be read only as "not contradicted by this
sample," never as precise per-technique reliability scores.

## 2026-08-18: the student now runs in the live system

Previously the trained ModernBERT student existed only as a benchmark
artifact: `/analyze` called the teacher directly, on every message, at
~5s/message. `app/student.py` now runs the student first as a deliberately
high-recall gate, and the teacher is only woken when the student finds
something worth investigating (see `app/main.py`). The student has no notion
of verbatim spans, so it can never answer on its own. Layer 2's span-grounding
guarantee is untouched, since a "flagged" message still goes through the full
teacher + explain pipeline exactly as before.

`eval/gate_calibration.py` measures the trade-off against the gold set at
`GATE_THRESHOLD = 0.05`:

| threshold | scam recall | clean wake rate | overall wake rate |
|---|---|---|---|
| 0.05 (chosen) | 0.87 | 0.07 | 0.29 |

At this threshold, 87% of scam messages correctly wake the teacher. A miss here
means Layer 2 never runs for that message, which is the cost of the gate being
wrong. Only 7% of clean messages needlessly wake the teacher, a pure latency
cost with no correctness cost (the teacher would presumably also find nothing).
Only 29% of all gold messages reach the teacher at all, so roughly 71% of
traffic resolves on the fast, student-only path. The gate's own 0.87 recall is
higher than the teacher's held-out 0.43 per-technique macro-recall would
suggest is achievable, because "does *anything* fire" (message-level) is an
easier bar than "does the *correct* technique fire" (per-technique). The gate
only needs to decide whether the teacher is worth calling; it doesn't need to
get the technique right.

This surfaced a deployment gotcha. The first call to the student model in a
fresh process (first-time `torch`/`transformers` import, weight load, GPU
transfer) measured ~20-30s. Loading it lazily on the first request would have
meant whichever user sent the first message after a server restart ate that
delay. A FastAPI `lifespan` startup hook (`app/main.py`) now calls
`app.student.warm_up()` before the server accepts any request, so the cost is
paid once at boot. Confirmed live: after warm-up, back-to-back requests settle
to 125-203ms, consistent with the 65-175ms range `eval/gate_calibration.py`
measures offline and nowhere near the 20-30s cold figure.

## 2026-08-18 update: held-out threshold calibration changes the ranking

The numbers below were originally reported in-sample: per-technique thresholds
were tuned by sweeping a grid against the same 250-message set they were then
scored on (`eval/metrics.py`'s old `per_technique_metrics`). Caveat #1 below
flagged this at the time: "every number here is an upper bound."
`eval/metrics.py` now also has `per_technique_metrics_cv`, a 5-fold threshold
calibration where each fold's threshold is chosen on the *other* folds only,
so every message is scored by a threshold that never saw its label.
`eval/compare.py` and `eval/run_eval.py` now print both. The held-out numbers
are the ones to use. The in-sample numbers are kept below only as the visible
measure of how much the old methodology overstated things.

Both the absolute numbers and the ranking change. In-sample, the distilled
student looked like it matched or beat the teacher (0.52 vs 0.56). Held-out,
that reverses and the teacher is back in front.

| Model | macro-F1 (in-sample) | macro-F1 (held-out) | latency/msg | size |
|---|---|---|---|---|
| Baseline (TF-IDF + LR) | 0.49 | 0.41 | ~0ms | negligible |
| Teacher (qwen2.5:7b, Ollama) | 0.53 | 0.51 | 4966ms | ~4.5GB |
| Student (distilled ModernBERT) | 0.56 | 0.48 | ~65-175ms | 574MB / 150M params |

(In-sample teacher macro-F1 above is 0.53, against the 0.52 originally
reported. This run used a live teacher pass, and the 7B model isn't perfectly
deterministic call to call even at low temperature, so a ~0.01 wobble on
re-running is expected and is no regression.)

A latency correction from the same day (2026-08-18), caught before the number
shipped anywhere else: an earlier pass through this same 250-message set
measured student latency at 14ms/msg via `eval/compare.py`'s
`student_predict`, run immediately after ~20 minutes of continuous teacher
inference on the same GPU. That number was a measurement artifact. The student
inherited boosted GPU clocks it doesn't sustain on its own, so the figure was
not steady-state. Repeated clean measurements (fresh process, no preceding GPU
load) of the same 250 messages land in the 65-175ms/msg range. `nvidia-smi`
confirms this laptop's RTX 4060 idles at 1890MHz against a 3105MHz boost
ceiling and doesn't hold a boost clock for one-message-at-a-time inference,
which is how a live product calls it. The table above and all latency claims
below use the corrected range.

The student is a good deal with a cost attached. Distillation buys a 28-76x
latency reduction (65-175ms vs the teacher's 4966ms; an earlier measurement
implied an erroneous 355x) and a ~8x smaller footprint (150M vs ~7B params)
for a modest 0.03 macro-F1 cost versus the teacher, and the student still
clearly beats the TF-IDF baseline. That headline is more accurate than
"student ties or beats the teacher," and the stricter methodology confirmed it
instead of revealing it as false: the teacher's own in-sample-to-held-out drop
(0.53 to 0.51) was smaller than the student's (0.56 to 0.48). The student's
original number benefited more from in-sample overfitting on its low-support
techniques (see the `channel_switch` and `payment_irreversibility` sections
below, both revised with the held-out numbers).

The plan's stated risk was "the student may lose badly to the teacher on rare
techniques with few positives." That risk is now visible: the student's
held-out `payment_irreversibility` collapses to F1=0.00 (a misleadingly
decent-looking 0.24 in-sample), and `channel_switch` drops from 0.24 to 0.12.
Both are discussed below. The two methodology caveats from the original
write-up still apply and now compound with the held-out result:

1. ~~Per-technique thresholds are tuned in-sample~~ (fixed above). It's worth
   remembering why it mattered this much: with 250 messages and 5-fold CV, each
   fold's threshold is chosen from ~200 examples, and for a 3-11-support
   technique that's only 2-9 positives to tune against per fold. The held-out
   numbers avoid the in-sample bias but are unstable: rerunning with a
   different seed would move the low-support rows more than the well-supported
   ones.
2. 7 of 11 techniques have single-digit to low-double-digit support in the
   250-message gold set (`sunk_cost_pressure`=3, `trust_transfer`=4,
   `payment_irreversibility`=4, `fake_scarcity`=5, `verification_theater`=8,
   `isolation`=11, `channel_switch`=12). Macro-averaging weights these exactly
   as heavily as `manufactured_urgency` (36) or `false_authority` (33), so a
   single misprediction on a 3-support class swings its F1 by 0.33+ and drags
   the macro average with it. Read the per-technique rows as well as the macro
   line.

## Original headline numbers (in-sample, kept for comparison)

Baseline/teacher/student numbers below are from the 2026-08-05 run of
`eval/compare.py`, after fixing a train/eval leakage bug in `eval/distill.py`.
The student had been trained on 190 of the 250 gold messages and was retrained
clean (see `eval/distill.py`'s `load_training_data` docstring).

| Model | macro-F1 | macro-P | macro-R | latency/msg | size |
|---|---|---|---|---|---|
| Baseline (TF-IDF + LR) | 0.49 | 0.44 | 0.62 | ~0ms | negligible |
| Teacher (qwen2.5:7b, Ollama) | 0.52 | 0.82 | 0.43 | 8120ms | ~4.5GB |
| Student (distilled ModernBERT) | 0.56 | 0.51 | 0.74 | 65ms | 574MB / 150M params |

## Concrete failure: the teacher is blind to `trust_transfer`

The teacher's `trust_transfer` row is P=0.00, R=0.00, F1=0.00 (N=4), a total
failure that no threshold choice explains. `classify()` was re-run live on all
4 gold `trust_transfer` positives to confirm:

| id | message (truncated) | true labels | teacher output |
|---|---|---|---|
| IN-016 | "Mom I lost my phone, this is my friend's number. Can you quickly send Rs 15000..." | trust_transfer | `reciprocity_hook` (0.9) |
| IN-038 | "Dad don't tell mom I called, but I'm stuck and need Rs 20,000 transferred..." | trust_transfer, isolation | `isolation` (1.0) only |
| IN-017 | "Hi it's me, changed my number after the SIM issue... don't call the old number" | trust_transfer, manufactured_urgency, channel_switch | *(nothing)* |
| uci-03864 | "Oh my god! I've found your number again! I'm so glad, text me back..." | trust_transfer | *(nothing)* |

IN-016 is the textbook "Hi Mum" scam, about as clean an example of
impersonating a specific trusted person as this taxonomy has, and the 7B
teacher both missed it and actively mislabeled it as `reciprocity_hook`. The
model appears to have no notion of "this message is pretending to be someone
the reader already knows," as distinct from `false_authority` (impersonating an
institution) or `isolation` (the secrecy instruction that often rides alongside
it, and which the teacher does pick up in IN-038). The student does somewhat
better on this technique (F1=0.46) but is far from solving it (see below).

## Concrete failure: the student barely learned `channel_switch`

Student `channel_switch` held-out: F1=0.12, P=0.25, R=0.08 (N=12). This is the
student's weakest technique and the one place the plan's predicted "student
loses to teacher on rare techniques" risk shows up (teacher held-out F1=0.38,
clearly ahead here). The in-sample number (F1=0.24) was already the worst
student row before the held-out fix. Held-out is worse still because the mean
decision threshold (0.21) hides fold-to-fold threshold instability. With only
13 positive training examples spread across 5 folds, whichever fold gets an
unlucky subset of those 13 for its threshold-selection step ends up with a
threshold that misses most of that fold's true positives. Misses, with the
student's raw probability (unaffected by which threshold is chosen; only the
pass/fail cutoff changed under CV):

- `uci-02941` ("You have 1 new message. Please call 08712400200."): proba 0.00
- `IN-012` ("...we cannot continue on a recorded government line. Please call me back on this personal WhatsApp number instead."): proba 0.06
- `IN-017` (the same "changed my number" message as above): proba 0.03

Only 13 of the 828 student training examples are positive for `channel_switch`
(see the `load_training_data` support printout). That is almost certainly the
cause: there isn't enough signal in training for the model to generalize the
pattern ("move off this channel") beyond a few surface phrases, so it defaults
to near-zero confidence on held-out paraphrases of the same idea.

## Concrete failure: `payment_irreversibility` over-fires on generic "Rs + urgency" text

Student `payment_irreversibility` held-out: P=0.00, R=0.00, F1=0.00 (N=4).
That is a complete collapse from the in-sample P=0.14/R=0.75/F1=0.24
originally reported here, and the biggest move of any number in this analysis
under the methodology fix. A moved threshold doesn't explain it: cost=47 with 0
true positives means the held-out predictions include false positives (the
over-firing behavior described below) while catching *none* of the 4 true
positives, across every fold. That signals the raw probabilities themselves
don't reliably separate the true positives from the noise, a deeper problem
than an in-sample threshold that happened to catch them by luck. Where
`channel_switch` under-fires, this technique over-fires: the model appears to
have learned "mentions a Rupee amount + urgency/fee language" as a loose proxy
for the technique in place of its definition (a payment *framed as a receipt*,
such as a UPI collect request or "scan to receive" trick). False positives
include messages whose real technique is `sunk_cost_pressure`,
`fake_scarcity`, or `false_authority`: different manipulation entirely, which
happens to also be money-shaped. The probabilities below are raw model output,
unaffected by which threshold is applied:

- `IN-035`: "You've already paid the Rs 12,000 registration fee... pay the final Rs 8,000..." (true: `sunk_cost_pressure`), proba 0.34
- `IN-020`: "Only 3 slots left... Deposit Rs 10,000 now..." (true: `fake_scarcity`), proba 0.20
- `IN-041`: "Please switch to Google Meet..." (true: `channel_switch`), proba 0.22

More concerning for a product that must never render "safe": it also
false-positives on entirely benign transactional messages. `IN-B01` (a
legitimate Amazon OTP notice, proba 0.17), `IN-B06` (an electricity bill
reminder, proba 0.15), `IN-B15` (a Myntra refund confirmation, proba 0.20), and
`IN-B10` (a bank's own PSA telling users never to share their PIN/OTP, proba
0.28, which also nearly trips `channel_switch` at 0.28). The model is picking
up "Rs amount + payment verb" as a surface cue that real scam messages,
legitimate bank notices, *and* anti-scam advisories all share. `IN-B10` is the
most interesting case: a message *warning people about scams* gets flagged
because it uses the same vocabulary scams use. With only 11 positive training
examples for this technique, the model doesn't have enough contrast between
"payment framed as receipt" and "any message involving a payment" to draw a
tighter boundary.

The same conflation shows up in the teacher. Testing `IN-043` ("you've already
paid Rs 60,000... final Rs 25,000 embassy fee or lose it") live, with true
labels `manufactured_urgency`, `false_authority`, `fear_of_consequence` and
`sunk_cost_pressure`, the teacher predicted `payment_irreversibility` (0.9) and
nothing else. It missed all four true labels and reached for the same "money
mention → payment_irreversibility" shortcut. So this is more than a
student-distillation artifact: the taxonomy itself looks hard to disambiguate
from money-adjacent language alone, for both model sizes.

## Concrete failure: `trust_transfer` vs `isolation`/`false_authority` confusion

The gold-labeling legend (`eval/export_labeling_sheet.py`'s Legend sheet)
explicitly calls out `false_authority` (impersonates an institution) vs
`trust_transfer` (impersonates a specific known person) as a common human
mix-up. The student reproduces the same confusion mechanically. Its
`trust_transfer` false positives are almost all `isolation` +
`false_authority` messages (police/officer-impersonation "stay on the call,
don't tell your family" scripts): `IN-004`, `IN-003`, `IN-034`, `IN-028`,
`IN-008`. None of them involve impersonating a specific known person. The
model seems to have partly learned "secrecy instruction + authority claim" as
a package deal that includes `trust_transfer`, when this taxonomy distinguishes
the two by *who* is being impersonated (an institution vs. someone specific to
the reader).

## What this means

> **Superseded in two places by the 2026-09-09 re-runs at the top of this
> file.** The teacher's held-out macro-F1 is 0.56, up from the 0.51 quoted
> below, so the student trails by 0.08 instead of 0.03. Every latency figure
> and speed multiple below should be read as unsupported: the same measurement
> spans a 9.5x range across runs on this hardware. The support-driven split
> described below still holds. The claims below are left as they were, without
> updates, because the point of this file is the record of what was believed
> when.

- The distillation thesis holds up under a clean, leakage-free, held-out eval,
  as a latency/size trade with an accuracy cost. The 574MB / ~65-175ms student
  trails the 4.5GB / 4966ms teacher by 0.03 macro-F1 (0.48 vs 0.51) once
  thresholds are calibrated properly, reversing the earlier in-sample result
  that had the student ahead. That is a different story from before, and still
  a strong resume artifact: 28-76x faster (see the latency correction above;
  an earlier "355x" figure was a measurement artifact) and ~8x smaller for a
  small accuracy cost. That cost has a clean support-driven split rather than a
  uniform quality gap across the taxonomy. On the 4 highest-support techniques
  (`manufactured_urgency`=36, `false_authority`=33, `reciprocity_hook`=28,
  `fear_of_consequence`=24) the student clearly beats the teacher (e.g.
  `fear_of_consequence` 0.88 vs 0.55). On the 7 lower-support techniques
  (support 3-12) the teacher wins 5 of 7, loses only `trust_transfer` (where
  it's completely blind, see above) and roughly ties `isolation`. In short, the
  student is the better choice on the techniques with enough training data, and
  the teacher's edge is concentrated where the student had too few positive
  examples to generalize from. That makes this a distillation story about data;
  the smaller architecture isn't categorically worse.
- Neither model is close to reliable on low-support techniques.
  `channel_switch`, `payment_irreversibility`, `sunk_cost_pressure`,
  `trust_transfer` and `fake_scarcity` all have single-digit to low-teens
  support in gold and correspondingly shaky, sometimes contradictory,
  per-technique behavior. This looks like a corpus-size problem more than a
  model-choice problem: a 7B LLM with a hand-written prompt and a from-scratch
  fine-tune both fail the same techniques (`trust_transfer`,
  `payment_irreversibility`) in overlapping ways.
- False positives on benign transactional/advisory messages (`IN-B01`,
  `IN-B06`, `IN-B10`, `IN-B15`) are the finding to take most seriously for a
  "never say safe" product. The failure mode has been observed: an anti-scam PSA
  gets partially flagged as itself manipulative.
- Macro-F1 alone is a misleading single number here, given how much of it is
  driven by 3-12-support classes. Any report of this eval should lead with the
  per-technique table and never quote the macro line in isolation.

## Layer 2: validator pass rate (found broken, then fixed)

This file is about Eval A (the classifier), but the plan's stated bigger
technical risk was Layer 2: the LLM-written purpose sentence that quotes the
manipulated span verbatim and explains why it's being used. Risk #2 in the plan
says explicitly: *"if the validator pass-rate is poor, report the number."*

It was poor: 0%. Checking `study/stimuli_cache.json` (the frozen treatment-arm
text for the human study) found all 7 explanation cards were table-only
fallback text, with none LLM-generated. Root cause: the old schema asked the
model to do two things in one free-text field, quoting the span verbatim *and*
explaining its purpose. Live re-generation showed the model almost never did
the first part (6/7 cases: "missing verbatim span"). It paraphrased the
message instead of copying it, even with an explicit instruction and a 40-word
budget. The prompt's own example didn't demonstrate a quote either, which
likely reinforced the pattern.

The fix narrowed the model's job to the purpose clause only and inserts the
verbatim quote programmatically from the Layer-1 span. This is the same
"guarantee by construction, not by hoping a validator catches a drift"
approach already used for the reality-check/counter-action text (see
`app/explain.py`'s module docstring). Re-verified live against all 60
`handcrafted_india` messages after the fix: 59/59 cards (100%) passed
validation on the first attempt, with 0 retries and 0 fallbacks. The 60th
message hit the Layer-1 out-of-range-confidence bug during `classify()`, so no
detections were available to test for it; that bug is now fixed (see the next
paragraph). `study/stimuli_cache.json` has been rebuilt with generated
explanations for all 4 scam stimuli.

The out-of-range-confidence bug was fixed on 2026-08-18. The teacher
occasionally emits a `confidence` outside [0, 1], because schema-constrained
decoding enforces JSON shape but not declared numeric range, and this used to
raise an uncaught `pydantic.ValidationError`. In `eval/compare.py`'s live
teacher pass, it showed up as gold messages "erroring" and being counted as
all-zero predictions, which silently undercounted teacher recall. In the
FastAPI app it would have been an unhandled 500 on `/analyze`. `app/llm.py`'s
`generate_structured` now clamps a pure range violation to its declared bound
and logs it instead of raising. Confirmed live in the same run that produced
the held-out numbers above: the teacher pass over all 250 gold messages clamped
2 out-of-range confidences and had 0 messages error out. Before the fix, this
class of bug silently cost a few gold messages' worth of teacher recall on
every eval run.

This is the more consequential finding of the two evals. The classifier being
mid (held-out macro-F1 0.41-0.56 across baseline/student/teacher) is a
defensible outcome, pre-registered as expected given corpus size. Layer 2
silently degrading to boilerplate on 100% of the study's treatment-arm content
would have invalidated the human study's entire premise. Eval B tests whether
the named, purpose-explained technique changes behavior compared with a bare
red-banner flag, and a fallback sentence like `"X" is a sign of Y.` is barely
different from a flag with a label attached.

## Not covered here

- ~~Inter-annotator agreement (Cohen's kappa)~~: done, see the 2026-08-23
  update above. Still open: it's only measured on an 80-message subset, and
  only against `labels_primary.jsonl`'s judgment calls. For example, whether
  `IN-B10` is really zero-technique isn't itself re-litigated; the kappa run
  only measures how consistently a second person applies the same taxonomy.
- ~~A held-out threshold-calibration split~~: done, see the 2026-08-18 update
  above.
