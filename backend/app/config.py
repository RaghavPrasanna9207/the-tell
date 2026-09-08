"""Runtime configuration, read from the environment.

Until this file existed nothing in the project read an environment variable
anywhere — every host, model name, path and threshold was a module-level
constant. That was fine while the only deployment target was "this laptop",
and it stops being fine the moment the same image has to run against a
container-networked Ollama, or against a hosted endpoint with a secret.

Every default here reproduces the previous hardcoded constant exactly, so a
checkout with no environment set behaves identically to before.

Nothing here reads a `.env` file: the values come from the process
environment, which is what Docker Compose, an HF Space secret, and a shell
export all produce. One less dependency and one less place for a key to end
up on disk.
"""

import os
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    """Empty string is treated as unset — an env var set to "" in a compose
    file or a Space secret is almost always an accident, and silently using
    it as a hostname produces a confusing failure much later."""
    return os.environ.get(name, "").strip() or default


def _env_float(name: str, default: float) -> float:
    raw = _env(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


# --- which teacher backend answers Layer 1 and Layer 2 ---------------------
# "ollama" = local Ollama server (the default, and what the eval numbers in
# the README were measured against). "nim" = NVIDIA NIM's hosted endpoint,
# which is what the public deployment uses because a 7B model on a free CPU
# host would be unusably slow.
LLM_BACKEND = _env("LLM_BACKEND", "ollama").lower()

# --- ollama ----------------------------------------------------------------
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
TEACHER_MODEL = _env("TEACHER_MODEL", "qwen2.5:7b-instruct-q4_K_M")

# --- nim -------------------------------------------------------------------
NIM_BASE_URL = _env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = _env("NIM_MODEL", "meta/llama-3.3-70b-instruct")
NIM_API_KEY = _env("NIM_API_KEY", "")

# --- student gate ----------------------------------------------------------
STUDENT_MODEL_DIR = Path(_env("STUDENT_MODEL_DIR", str(_BACKEND_DIR / "models" / "student")))
STUDENT_HF_REPO = _env("STUDENT_HF_REPO", "")
"""Hugging Face repo to pull the distilled student from when STUDENT_MODEL_DIR
is empty — the deployed image has no local training output. Blank means "no
remote fallback", which keeps a fresh clone's behavior unchanged: no student
means no gate, and every message goes straight to the teacher."""

GATE_THRESHOLD = _env_float("GATE_THRESHOLD", 0.05)

# --- api -------------------------------------------------------------------
CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

STATIC_DIR = Path(_env("STATIC_DIR", str(_BACKEND_DIR / "static")))
"""Built frontend to serve from the same origin as the API. Only mounted if
it exists, so local development (Vite on :5173, API on :8000) is unaffected."""


def active_teacher_model() -> str:
    """The model name whichever backend is selected will actually call.
    Reported in evals and logs so a number is never ambiguous about which
    teacher produced it."""
    return NIM_MODEL if LLM_BACKEND == "nim" else TEACHER_MODEL
