# Deploying to Hugging Face Spaces

The public demo runs the **full cascade** — the distilled student gates on the
Space's CPU, and only messages that clear the gate reach the hosted teacher.
That was the sticking point that kept this project undeployed: the product is
the teacher's named-technique explanations, and a 7B model on a free CPU host
would have meant shipping a degraded, gate-only slice of it. Moving the teacher
to a hosted endpoint removes that constraint without weakening the demo.

Free tier: Docker SDK, 2 vCPU / 16GB RAM, no credit card. Sleeps after 48h idle.

## One-time setup

**1. Publish the student.** `backend/models/` is gitignored, so the image has
nothing to load without this.

```bash
huggingface-cli login
python backend/eval/publish_student.py --repo <user>/the-tell-student-modernbert
```

**2. Get a NIM API key** — https://build.nvidia.com, free, no credit card.

**3. Create the Space** at https://huggingface.co/new-space with **Docker** as
the SDK (blank template), then in Settings → Variables and secrets:

| Name | Kind | Value |
|---|---|---|
| `NIM_API_KEY` | **Secret** | `nvapi-...` |
| `STUDENT_HF_REPO` | Variable | `<user>/the-tell-student-modernbert` |

`NIM_API_KEY` must be a *secret*, not a variable — variables are visible to
anyone who can view the Space.

**4. Push.** The Space repo needs `Dockerfile` at its root and a `README.md`
carrying the front matter below. Build context is the repo root, so both
`backend/` and `frontend/` must be present.

```bash
git remote add space https://huggingface.co/spaces/<user>/the-tell
git push space HEAD:main
```

## Required Space README front matter

Hugging Face reads this from `README.md` at the Space root. Without
`app_port: 7860` it will not route traffic to the container.

```yaml
---
title: The Tell
emoji: 🎣
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

## How the image is put together

See `Dockerfile.space`. Three things in it are non-obvious:

- **CPU torch is installed explicitly** from the PyTorch CPU index. A plain
  `pip install torch` resolves to the CUDA build — ~2.5GB of GPU libraries that
  cannot run here.
- **The student is baked in at build time** via `snapshot_download`. The free
  tier has no persistent volume, so a runtime download would repeat on every
  cold start, and `/data` isn't available during build anyway.
- **The container runs as uid 1000.** Every `COPY` uses `--chown=user`;
  without it the app cannot read its own files.

The frontend is built in a first stage with `VITE_API_BASE=""` and served by
FastAPI from the same origin, so there is no CORS boundary. `app/main.py`
mounts it last, after `/health` and `/analyze`, because Starlette matches
routes in registration order and a catch-all mounted earlier would swallow the
API.

## Verifying a deploy

It is not done until all of these hold:

1. The Space URL loads the UI with no console errors.
2. Pasting a digital-arrest message returns technique cards with the
   manipulated span highlighted and **clickable** source links.
3. Pasting a benign message returns "No manipulation techniques detected" —
   and never the word "safe".
4. The gate is genuinely running: a clean message resolves without a NIM call.
   Confirm in the Space logs via the `gate: clean, teacher skipped` line.
5. Cold-start-from-sleep time is measured and written into the root README.
   The 20-30s warm-up figure recorded elsewhere was measured on an RTX 4060;
   two shared vCPUs will be slower and the real number belongs in the docs.
6. `NIM_API_KEY` appears only in Space secrets — never in git, never in the
   image, never in a build arg.
