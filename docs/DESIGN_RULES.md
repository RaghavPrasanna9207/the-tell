# Design rules

The constraints this codebase is built to hold. They're written down because
most of them are invariants a reader can't infer from any single file, and
several are enforced by tests rather than by convention.

## The model selects and phrases facts. It never invents them.

Every factual claim in an explanation — legal facts, what RBI does and
doesn't do, what real police do — comes from `backend/data/counter_moves.yaml`,
a hand-authored, source-cited table. The LLM's job in Layer 2 is retrieval,
phrasing, and nothing else.

This is enforced by construction, not by asking nicely. The reality-check and
counter-action text is inserted programmatically. So is the verbatim quote:
an earlier design asked the model to both quote the manipulated span verbatim
*and* explain its purpose in one field, and it almost never did the first
part — a 0% validator pass rate that went unnoticed because the fallback text
looked plausible. Narrowing the model's job to the purpose clause and
inserting the quote from the Layer-1 span made the guarantee hold
structurally. See `backend/app/explain.py` and `ERROR_ANALYSIS.md`.

## Never say "safe."

When no manipulation technique is detected, the output is "No manipulation
techniques detected" — absence of evidence, not evidence of absence. Nothing
in the API or UI may render or imply that a message is safe. `Analysis.is_clean`
exists, and the API layer is responsible for how it's phrased.

This matters more than it looks: the classifier misses real scams (see the
gate's measured recall in `eval/cascade_eval.py`), so "we found nothing" and
"there is nothing" are genuinely different statements.

## `taxonomy.py` is the single source of truth.

The 11 manipulation techniques and their Pydantic schema live in exactly one
place, `backend/app/taxonomy.py`, and are imported everywhere else: the JSON
schema fed to the constrained decoder, the FastAPI response model, the eval
label space, and `counter_moves.yaml`'s key set. The technique list is never
redefined elsewhere. `tests/test_taxonomy_integrity.py` fails loudly on an
orphan in either direction.

## Every `counter_moves.yaml` entry needs a source *and* a followable link.

No reality-check ships without a citation, and the citation must be a URL on
an authoritative domain — the cyber-crime authority, the central bank, or the
government press service. Tests enforce both.

The link requirement exists because prose citations alone ("I4C advisory on
digital arrest scams") can't be checked by a reader, and the entire product
rests on these facts being checkable. The URLs are institution- and
advisory-level, verified to resolve; their content is not machine-verified
against each claim, because these government sites block automated fetching.
Re-checking them by hand when the table changes is part of editing it.

## No raw message logging by default.

User-submitted message text is not persisted unless explicitly opted into
(the study harness only), and even then it is PII-scrubbed — phone numbers,
UPI IDs, account numbers, names — before being written to disk. Study
responses record a random unlinked participant id, arm, stimulus id, response
code, confidence, and timestamp. Nothing else.

## Report metrics honestly, including the ones that look bad.

Negative results, wide confidence intervals, small N, validator failure
rates, and student-vs-teacher gaps get reported as they are. Several numbers
in this repo got worse under scrutiny and were republished worse:
in-sample thresholds became held-out ones and reversed a ranking; a
355x latency claim turned out to be a GPU-clock artifact and became 28-76x;
a human-study hypothesis was withdrawn as unmeasurable after its instrument
turned out to be broken. That's the standard — not rounding up, and not
quietly keeping a good number whose basis has failed.

## Efficiency is the architecture

A 150M-parameter distilled student gates a much larger teacher. Roughly 71%
of messages resolve on the student-only path and never reach the teacher at
all. That's the point of the design: the expensive model is woken only when
something warrants it, which is what makes the system viable against a
rate-limited hosted endpoint as well as on a laptop.

The student has no notion of verbatim spans, so it can never answer on its
own — Layer 2's span-grounding guarantee only ever comes from the teacher.
The gate decides whether to ask, not what the answer is.

`eval/cascade_eval.py` measures what that gate costs end to end, because the
teacher's own score describes a component, not the shipped product.

## Scope boundaries

- English only. No i18n.
- Web only. No mobile app.
- No screenshot OCR — paste-in only.
- The frontend is intentionally thin (one page, ~180 lines). No design
  system, no routing, no state management library.

## Stack

- Backend: Python 3.12, FastAPI, Pydantic, `sklearn`, HuggingFace
  `transformers`/`Trainer`.
- Teacher: swappable via `LLM_BACKEND` (`app/config.py`). Local Ollama with
  `qwen2.5:7b-instruct-q4_K_M` is the default and the backend every eval
  number in the README was measured against; NVIDIA NIM serves the public
  deployment, where a 7B model on a free CPU host would be unusably slow.
  Both use JSON-schema-constrained decoding.
- Student: `answerdotai/ModernBERT-base`, distilled from teacher labels.
- Frontend: React + Vite + TypeScript, single page.

## Verification habits

- Every feature gets a `curl` or `pytest` check before being called done.
- Touching the taxonomy or the counter-move table means running
  `pytest backend/tests/` before moving on — a broken source-cite or an
  orphaned technique should fail loudly, not silently.
- An eval number is reported with the thing it was measured against. The
  gold set is 250 messages, 190 of them UCI SMS spam, with 45 of 69 positive
  examples author-written and author-labeled; any number scored against it
  carries that caveat.
