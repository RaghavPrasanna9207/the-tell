# ScamShield Explainer

A scam shield that explains the manipulation technique behind a message, not just flags it.

Spam detection is solved and crowded. The unsolved problem is that detection doesn't
change behavior — victims of digital-arrest and UPI scams are frequently told
"this is a scam" and proceed anyway, because a live human is applying pressure and
the app is a red icon. This project's bet: the product isn't classification, it's
counter-persuasion — naming the specific technique being used, grounded in
verifiable facts, in language a person under pressure will actually believe.

Fully local. No API key, no account, no cost — everything runs on-device via
Ollama + a distilled local classifier.

**Status:** taxonomy, both layers, gold labeling, distillation, three-way eval,
and the student cascade (below) complete. Human study in progress. See
`CLAUDE.md` for project rules and the full plan at
`~/.claude/plans/1-a-scam-shield-temporal-muffin.md`.

## Architecture

```
paste text ──> GATE: distilled ModernBERT student (app/student.py)
               high-recall pre-filter, ~65-175ms, no verbatim spans
                              │
               nothing clears the gate? ──> "no techniques detected", teacher never called
                              │ (something fired)
                              ▼
               Layer 1: multi-label technique classifier
               (Ollama/Qwen2.5-7B, schema-constrained)
                              │
               techniques[] + verbatim spans + confidence
                              ▼
               Layer 2: grounded explanation generator
               (retrieves facts from counter_moves.yaml — never invents them)
                              ▼
               Layer 3: break the isolation
               (one-tap trusted contact, 1930 helpline, share-to-verify)
```

The student is a real part of the running system, not just a benchmark
artifact: `/analyze` runs it first as a deliberately high-recall gate, and
only wakes the (accurate but ~5s/message) teacher when the student finds
something worth investigating — roughly 71% of real messages resolve on the
fast student-only path (measured against the gold set, see
`eval/gate_calibration.py` and `ERROR_ANALYSIS.md`). The student has no
notion of verbatim spans, so it can never be the thing that answers on its
own; Layer 2's span-grounding guarantee only ever comes from the teacher. If
the student model isn't present (a fresh clone before `eval/distill.py` has
been run), the gate is skipped and every message goes straight to the
teacher — same behavior as before this cascade existed.

## Setup

```bash
# Backend — use a venv. This project's teacher/distillation deps
# (transformers, torch) commonly clash with whatever else is in a shared
# global Python install (seen firsthand: a global triton/unsloth-zoo from
# an unrelated project broke transformers' import here) — an isolated venv
# avoids that entirely.
cd backend
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat for cmd.exe

# CUDA torch FIRST (plain `pip install torch` / requirements.txt gives you
# a useless CPU-only build on this machine's RTX 4060 — see the note in
# requirements.txt)
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"   # must print True

pip install -r requirements.txt

# Ollama (local LLM runtime) — install from https://ollama.com, then:
ollama pull qwen2.5:7b-instruct-q4_K_M

# Corpus — raw files are gitignored (large / re-downloadable), so build them:
curl -sL -o backend/data/corpus/raw/sms_spam_uci.zip \
  "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
unzip -o backend/data/corpus/raw/sms_spam_uci.zip -d backend/data/corpus/raw/
python backend/data/build_corpus.py

# Frontend
cd frontend
npm install
```

The Mendeley SMS Phishing dataset (`f45bkkt8pr`) has no stable direct-download
URL — grab it manually from
[data.mendeley.com](https://data.mendeley.com/datasets/f45bkkt8pr/1) and drop
the CSV into `backend/data/corpus/raw/` if you want the extra volume; the
pipeline works without it.

## Running

```bash
ollama serve &
cd backend && uvicorn app.main:app --reload
cd frontend && npm run dev
```

## Testing

```bash
cd backend && pytest        # 87 tests: taxonomy/counter-move integrity,
                             # classify/explain logic, metrics, cascade gate,
                             # full /analyze pipeline (mocked at the LLM boundary)
python eval/run_eval.py     # baseline classifier metrics against the corpus
```

## Evals

Two spines: classifier performance (baseline vs. teacher vs. distilled student,
cost-weighted for false negatives) and a small pre-registered human study testing
whether named-technique explanations change stated behavior compared to a plain
scam flag. Eval A results are below. See `study/PREREGISTRATION.md` for Eval B's
hypothesis and analysis plan, committed before any participant runs — Eval B
results and the full writeup land once the study completes.

```bash
# Silver-label ~1000 corpus messages with the local teacher (for distillation)
python backend/eval/silver_label.py

# Real distillation once silver labels exist (falls back to a smoke test otherwise)
python backend/eval/distill.py

# Three-way baseline/teacher/student comparison against the gold set
python backend/eval/compare.py

# Measure the cascade gate's scam-recall / teacher-wake-rate trade-off
python backend/eval/gate_calibration.py

# Get a second annotator (not a study participant — see below) to label the
# 80-message kappa subset, either via CLI:
python backend/eval/label.py --annotator second --kappa-subset 80
# ...or hand them a small spreadsheet instead of asking them to run Python:
python backend/eval/export_labeling_sheet.py --annotator second --kappa-subset 80
#   -> send them gold_labeling_second.xlsx, they fill it in, you get it back
python backend/eval/import_labeling_sheet.py --annotator second

# Cohen's kappa between labels_primary.jsonl and labels_second.jsonl
python backend/eval/kappa.py

# One-time: freeze Layer 1/2 output for the study's 6 fixed stimuli
python study/build_stimuli_cache.py

# One run per participant (randomly, balanced, assigned to control/treatment)
python study/run_study.py

# Fixed analysis plan against collected responses
python study/analyze_study.py
```

**Honesty note:** the numbers below run against the real 250-message hand-labeled
gold set (`backend/data/corpus/gold/labels_primary.jsonl`), single annotator —
no second annotator or Cohen's kappa yet. Per-technique thresholds are chosen via
5-fold cross-validation — each fold's threshold is picked using only the *other*
folds, so every message is scored by a threshold that never saw its own label
(`eval/metrics.py`'s `per_technique_metrics_cv`; `eval/compare.py` also prints the
simpler in-sample numbers for comparison, which run noticeably higher — that gap
*is* the finding, see `ERROR_ANALYSIS.md`). Full caveats and per-technique failure
analysis in `backend/eval/ERROR_ANALYSIS.md`.

### Results (three-way comparison, N=250 gold, held-out threshold CV)

| System | Macro-F1 | Macro-P | Macro-R | Latency/msg | Size |
|---|---|---|---|---|---|
| Baseline (TF-IDF + LR) | 0.41 | 0.33 | 0.55 | ~0ms (CPU) | negligible |
| Teacher (Qwen2.5-7B, live) | **0.51** | **0.75** | 0.43 | 4966ms | ~4.5GB |
| Student (distilled ModernBERT) | 0.48 | 0.42 | **0.60** | **~65-175ms** | **574MB / 150M params** |

Run via `eval/compare.py`, same 250 gold messages for all three, thresholds
calibrated held-out.

**A latency honesty note, because an earlier number here was wrong.** An
earlier measurement reported student latency as low as 14ms/msg — that turned
out to be a measurement artifact, not a real number: the student ran
immediately after ~20 minutes of continuous teacher inference on the same GPU,
inheriting boosted clock speeds it doesn't normally sustain. Repeated clean
measurements of the same 250-message pass, in a fresh process, land in the
**65-175ms/msg range** (confirmed via `nvidia-smi`: this laptop's RTX 4060 idles
at 1890MHz against a 3105MHz boost ceiling, and single bursty inference calls —
one message at a time, exactly how a live product is actually used — don't hold
it there). That's still a genuine 28-76x latency win over the teacher's
4966ms/msg, just not the inflated 355x the anomalous measurement implied. Not
rounding up a good number is the whole point of this project's honesty
standard, so this gets corrected in the open rather than quietly kept.

**Read this as a latency/size trade, not a free accuracy win.** An earlier,
in-sample-thresholded version of this table had the student slightly *ahead* of
the teacher (0.56 vs 0.52) — that ranking didn't survive fixing the threshold
methodology to be genuinely held-out. With honest calibration the teacher is
back in front, 0.51 vs 0.48. The gap has a clean, legible cause: on the 4
highest-support techniques (`manufactured_urgency`, `false_authority`,
`reciprocity_hook`, `fear_of_consequence` — support 24-36) the student clearly
*beats* the teacher; the teacher's edge is concentrated in the 7 techniques with
single-digit-to-low-double-digit support (as low as 3), where the student had too
few positive training examples to generalize. That's a genuinely better resume
story than the original number — distillation buys a large, real latency win and
~8x smaller footprint for a small, well-understood accuracy cost, not a gap
that's hidden or hand-waved. Full per-technique breakdown in `ERROR_ANALYSIS.md`.

**The distillation quality gap, and what actually fixed it.** The first real
distillation run scored macro-F1 = **0.10**, with zero recall on 8 of 11
techniques. Root cause: the ~940-message silver-labeled training pool had almost
no positive examples for most techniques, because the corpus behind it is mostly
generic UCI SMS spam — it just doesn't contain India-specific digital-arrest/UPI
manipulation language. The fix was to write more training data, not tune the
training loop: 78 new hand-labeled messages targeting exactly the
underrepresented techniques (`backend/data/corpus/raw/handcrafted_india_train.jsonl`
— training-only, never touches the gold-eval set). Re-running distillation on the
augmented pool took gold-holdout macro-F1 from 0.10 to 0.47 (later 0.56 in-sample
once a train/eval leakage bug was fixed — see `eval/distill.py`'s
`load_training_data` docstring — and 0.48 under the held-out threshold
calibration in the table above), and every technique moved off zero recall.

**The teacher is blind to `trust_transfer`** (impersonating a specific known
person, e.g. "Mom, I lost my phone") — P=R=F1=0.00 on all 4 gold positives,
including one live-reran case where it actively mislabeled the textbook "Hi Mum"
scam as `reciprocity_hook`. The student does better (held-out F1=0.33) but the
technique stays hard for both models. **False positives on benign
transactional/advisory messages** (a legitimate bank OTP, an electricity bill
reminder, a bank's own anti-scam PSA) are the finding to take most seriously for
a "never say safe" product — the student's held-out `payment_irreversibility`
row is F1=0.00 (0 true positives caught across all 5 folds) precisely because the
model picks up "Rs amount + payment verb" as a surface cue shared by real scams
and legitimate messages alike, rather than the technique's actual definition.
Full breakdown, with message-level examples, in `ERROR_ANALYSIS.md`.

Three real bugs surfaced along the way, all fixed. The Layer 2 validator checked
that a technique's span appeared verbatim but not that it was actually *quoted*,
so one live explanation came out as a run-on, ungrammatical sentence (now
checked — see "Layer 2 validator pass rate" below). Per-technique thresholds were
originally tuned and scored on the same 250-message set — an in-sample upper
bound rather than a genuine estimate — fixed by the 5-fold held-out calibration
described above. And the teacher occasionally emits a confidence value outside
[0, 1] (seen ~0.2–5% of calls) — schema-constrained decoding enforces JSON
*shape*, not numeric *bounds* — which used to raise an unhandled 500 on
`/analyze`; `app/llm.py`'s `generate_structured` now clamps a pure range
violation to its declared bound and logs it, confirmed live (2 clamped, 0 errors
across the 250-message teacher pass behind the table above).

**Layer 2 validator pass rate.** Initially 0% — `study/stimuli_cache.json`'s
frozen treatment-arm text was all table-only fallback, not real LLM output,
because the old schema asked the model to both quote the manipulated span
verbatim and explain its purpose in one field, and it almost never did the
first part. Fixed by narrowing the model's job to the purpose clause only and
inserting the verbatim quote programmatically from the Layer 1 span. Re-verified
against all 60 `handcrafted_india` messages after the fix: 59/59 cards (100%)
passed validation on the first attempt, 0 retries, 0 fallbacks.
