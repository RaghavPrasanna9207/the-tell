# Frontend

Single-page React + TypeScript + Vite client for The Tell. One screen: paste a
message, get back named-technique cards with the manipulated span highlighted
in place and a followable citation per card.

Intentionally thin — one component, hand-written CSS, no design system, no
router, no state management library. See `docs/DESIGN_RULES.md`.

```bash
npm install
npm run dev      # http://localhost:5173, expects the API on :8000
npm run build    # tsc -b && vite build  (the typecheck runs here, not separately)
npm run lint     # oxlint
```

## Configuration

`VITE_API_BASE` sets the API origin at build time.

- Unset → `http://localhost:8000` (local development against `uvicorn`).
- Empty string → same-origin, which is what the deployed build uses: FastAPI
  serves this bundle, so there's no cross-origin hop and no CORS to configure.

## Two things worth knowing before editing

**Never render "safe."** When the API returns `is_clean: true` the copy is "No
manipulation techniques detected" plus an explicit line that this does not mean
the message is safe. The classifier misses real scams — see
`backend/eval/cascade_eval.py` for the measured miss rate — so "we found
nothing" and "there is nothing" are different statements and the UI must not
collapse them.

**The highlight is a substring match, not a model output.** `highlightSpan()`
does `text.indexOf(span)` and renders `<mark>`. It can do that safely because
the backend drops any span that isn't a verbatim substring of the input
(`backend/app/classify.py`), so a hallucinated quote never reaches this
component. If that check ever moves, this breaks silently — it would just fail
to highlight rather than error.
