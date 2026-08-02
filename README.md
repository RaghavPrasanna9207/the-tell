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

**Honesty note:** the eval numbers you'll see today from `run_eval.py` run against
a ~60-message draft-labeled set (self-labeled at authoring time, no second
annotator, no kappa) — not the ~250-message hand-labeled gold set with
inter-annotator agreement the plan calls for. Treat current numbers as a pilot,
not a result; the harness itself is real and reusable once real gold labels land.
