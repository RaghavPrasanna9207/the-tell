# Backend image for local development via docker compose.
# The public deployment uses deploy/Dockerfile.space instead, which also
# builds and serves the frontend from the same container.

FROM python:3.12-slim

# curl is used by the compose healthcheck; nothing at runtime needs it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch, from the CPU index, BEFORE requirements-serve.txt. A plain
# `pip install torch` resolves to the CUDA build: ~2.5GB of GPU libraries
# that cannot run in this container. The student is a 150M-param model and
# runs acceptably on CPU — see app/student.py's device selection.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app/ ./app/
COPY data/ ./data/

# Defaults suit compose; every one is overridable (see app/config.py).
ENV LLM_BACKEND=ollama \
    OLLAMA_HOST=http://ollama:11434 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# No --reload: that's a development-server flag, and it would also run the
# student warm-up twice (once per worker process).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
