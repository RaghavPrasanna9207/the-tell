# ScamShield Explainer — Project Rules

A scam shield that explains the manipulation technique behind a message, not just flags it. Thesis: detection doesn't change behavior — counter-persuasion might. Built as a resume/interview artifact for an AI engineering role.

Full plan: `C:\Users\Raghav Prasanna\.claude\plans\1-a-scam-shield-temporal-muffin.md`

## Non-negotiables

**Zero cost. Fully local. No API, no account, no key, ever.**
All inference runs through Ollama on local hardware (RTX 4060, 8GB VRAM). Never add a cloud LLM call, never add a `.env` for an API key, never suggest "just use the API for this part." If a local approach seems harder, do the local approach anyway — that difficulty is the point of the project.

**The model selects and phrases facts. It never invents them.**
Every factual claim in an explanation (legal facts, what RBI does/doesn't do, what real police do) must come from `backend/data/counter_moves.yaml`, a hand-authored, source-cited table. The LLM's job in Layer 2 is retrieval + phrasing + verbatim-span-quoting — never generation of new factual claims. This is enforced by a validator, not just a prompt instruction. Do not weaken this to make the model's job easier.

**Never say "safe."**
When no manipulation technique is detected, the output is "No manipulation techniques detected" — absence of evidence, not evidence of absence. Never render or imply "this message is safe."

**No raw message logging by default.**
User-submitted message text is not persisted unless explicitly opted into (study harness only), and even then must be PII-scrubbed (phone numbers, UPI IDs, account numbers, names) before being written to disk.

**`taxonomy.py` is the single source of truth.**
The 11 manipulation techniques and their Pydantic schema live in exactly one place (`backend/app/taxonomy.py`) and are imported everywhere else: the JSON schema fed to the Ollama constrained decoder, the FastAPI response model, the eval label space, and `counter_moves.yaml`'s key set. Never redefine the technique list elsewhere.

**Every `counter_moves.yaml` entry needs a `source`.**
No reality-check ships without a citation (I4C, RBI, or the 1930 helpline). A test enforces this — don't bypass it.

## Scope boundaries (do not expand without being asked)

- English only. No i18n.
- Web only. No Android/mobile app.
- No screenshot OCR (cut for scope — paste-in only).
- Frontend is intentionally thin (~150 lines, one page). Do not add a design system, routing, or state management library.

## Stack

- Backend: Python 3.12, FastAPI, Pydantic, `sklearn`, HuggingFace `transformers`/`Trainer`.
- LLM runtime: Ollama, teacher model `qwen2.5:7b-instruct-q4_K_M` (fallback `llama3.1:8b-instruct-q4_K_M`), constrained decoding via the `format` JSON-schema parameter.
- Student model: `answerdotai/ModernBERT-base`, distilled from teacher labels, deployed for inference.
- Frontend: React + Vite + TypeScript, single page.
- No Docker, no k8s, no cloud infra of any kind — this runs on a laptop.

## Verification habits

- Every new feature gets a `curl` or `pytest` check before being called done — see "End-to-end verification" in the plan.
- When touching the taxonomy or counter-move table, run the integrity tests (`pytest backend/tests/`) before moving on — a broken source-cite or an orphaned technique should fail loudly, not silently.
- Report metrics honestly, including negative results (validator failure rate, student-vs-teacher gaps, N=20 being underpowered). Don't round up.
