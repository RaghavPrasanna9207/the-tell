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

**Status:** early build. See `CLAUDE.md` for project rules and the full plan at
`~/.claude/plans/1-a-scam-shield-temporal-muffin.md`.

## Architecture

```
paste text ──> Layer 1: multi-label technique classifier
               (Ollama/Qwen2.5-7B, schema-constrained ──distill──> ModernBERT-base)
                              │
               techniques[] + verbatim spans + confidence
                              ▼
               Layer 2: grounded explanation generator
               (retrieves facts from counter_moves.yaml — never invents them)
                              ▼
               Layer 3: break the isolation
               (one-tap trusted contact, 1930 helpline, share-to-verify)
```

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
cd backend && pytest        # 53 tests: taxonomy/counter-move integrity,
                             # classify/explain logic, metrics, full /analyze
                             # pipeline (mocked at the LLM boundary)
python eval/run_eval.py     # baseline classifier metrics against the corpus
```

## Evals

Two spines: classifier performance (baseline vs. teacher vs. distilled student,
cost-weighted for false negatives) and a small pre-registered human study testing
whether named-technique explanations change stated behavior compared to a plain
scam flag. See `study/PREREGISTRATION.md` for the study's hypothesis and analysis
plan, committed before any participant runs. Results and full methodology writeup
land once week 2–3 land.

```bash
# Silver-label ~1000 corpus messages with the local teacher (for distillation)
python backend/eval/silver_label.py

# Real distillation once silver labels exist (falls back to a smoke test otherwise)
python backend/eval/distill.py

# Three-way baseline/teacher/student comparison against the gold set
python backend/eval/compare.py

# One-time: freeze Layer 1/2 output for the study's 6 fixed stimuli
python study/build_stimuli_cache.py

# One run per participant (randomly, balanced, assigned to control/treatment)
python study/run_study.py

# Fixed analysis plan against collected responses
python study/analyze_study.py
```

**Honesty note:** every number below runs against a ~60-message draft-labeled set
(self-labeled at authoring time, no second annotator, no kappa) — not the
~250-message hand-labeled gold set with inter-annotator agreement the plan calls
for. Treat current numbers as a pilot, not a result; the harness itself is real
and reusable once real gold labels land.

### Results (three-way comparison, N=60 draft gold)

| System | Macro-F1 | Precision | Recall | Latency/msg | Size |
|---|---|---|---|---|---|
| Baseline (TF-IDF + LR) | 0.52 | 0.42 | 0.77 | ~0ms (CPU) | negligible |
| Teacher (Qwen2.5-7B, live) | **0.64** | 0.83 | 0.53 | 2358ms | ~4.5GB |
| Student (distilled ModernBERT) | 0.47 | 0.36 | 0.77 | **67ms** | **574MB** |

Run via `eval/compare.py`, same 60 messages for all three.

**The distillation quality gap, and what actually fixed it.** The first real
distillation run scored macro-F1 = **0.10**, with zero recall on 8 of 11
techniques. Root cause: the ~940-message silver-labeled training pool had almost
no positive examples for most techniques, because the corpus behind it is mostly
generic UCI SMS spam — it just doesn't contain India-specific digital-arrest/UPI
manipulation language. The one corpus subset that does (the 60 handcrafted
messages) had to be excluded from training entirely to keep the gold-eval
holdout honest, so the student trained on almost no in-domain positive examples
for 8 of the 11 techniques.

The fix was to write more training data, not tune the training loop: 78 new
hand-labeled messages targeting exactly the underrepresented techniques
(`backend/data/corpus/raw/handcrafted_india_train.jsonl` — training-only, never
touches the gold-eval set). Re-running distillation on the augmented pool took
the true gold-holdout macro-F1 from 0.10 to **0.47**, and every technique moved
off zero recall. `trust_transfer` is the one technique that stays weak across
*every* system (F1 = 0.0 for both baseline and teacher, support = 4) — that's a
genuine low-support problem, not something specific to distillation.

Two real bugs surfaced along the way, both fixed: the Layer 2 validator checked
that a technique's span appeared verbatim but not that it was actually *quoted*,
so one live explanation came out as a run-on, ungrammatical sentence (now
checked). And the teacher occasionally emits a confidence value outside [0, 1]
(seen ~0.2–5% of calls depending on the sample) — schema-constrained decoding
enforces JSON *shape*, not numeric *bounds*, which is exactly why the Pydantic
validation step in `app/llm.py` isn't optional scaffolding.
