"""FastAPI app. Wires /analyze to Layer 1 (classify) + Layer 2 (explain).

Layer 1 runs as a two-stage cascade rather than calling the teacher
directly: the distilled ModernBERT student (app.student) screens every
message first as a deliberately high-recall gate, and the teacher is only
woken when the student finds something worth investigating. This is what
makes the distilled student a real part of the running system rather than a
benchmark-only artifact, and it's what makes a rate-limited hosted teacher
viable — most messages never cost a request. eval/cascade_eval.py measures
what the gate costs in accuracy. If the student model isn't available (a
fresh clone that has never run eval/distill.py, with no STUDENT_HF_REPO
configured), the gate is skipped and every message goes straight to the
teacher.

Which teacher answers is app.config.LLM_BACKEND — local Ollama or hosted
NIM. If it isn't reachable or usable, /analyze returns a 503 with an
actionable message; it never falls back to fabricated output. An
explanation invented because the model was down would be worse than no
explanation. See docs/DESIGN_RULES.md.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import config
from app.classify import classify
from app.explain import explain_all
from app.llm import LLMUnavailableError
from app.student import is_available as student_available, should_investigate, warm_up
from app.taxonomy import Technique

logging.basicConfig(level=logging.INFO, format="%(message)s")  # no-op if root already has a handler
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading the student (first-time torch/transformers import + weights +
    # GPU transfer) costs ~20-30s cold — pay that once here, not on whichever
    # user's request happens to be first.
    start = time.monotonic()
    warm_up()
    if student_available():
        logger.info("student gate warmed up (%.0fms)", (time.monotonic() - start) * 1000)
    yield


app = FastAPI(title="The Tell", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # defaults to the Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class TechniqueCard(BaseModel):
    technique: Technique
    plain_name: str
    span: str
    confidence: float
    explanation: str
    source: str
    source_url: str


class AnalyzeResponse(BaseModel):
    is_clean: bool
    """True means no manipulation technique was detected. Never render this
    as "safe" in the UI — absence of evidence, not evidence of absence."""
    cards: list[TechniqueCard]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    start = time.monotonic()

    if student_available() and not should_investigate(req.text):
        logger.info("gate: clean, teacher skipped (%.0fms)", (time.monotonic() - start) * 1000)
        return AnalyzeResponse(is_clean=True, cards=[])

    try:
        classify_result = classify(req.text)
        if classify_result.analysis.is_clean:
            logger.info("gate: investigated, teacher found nothing (%.0fms)", (time.monotonic() - start) * 1000)
            return AnalyzeResponse(is_clean=True, cards=[])

        explanations = explain_all(classify_result.analysis.detections)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    cards = [
        TechniqueCard(
            technique=result.detection.technique,
            plain_name=result.counter_move.plain_name,
            span=result.detection.span,
            confidence=result.detection.confidence,
            explanation=result.explanation,
            source=result.counter_move.source,
            source_url=result.counter_move.source_url,
        )
        for result in explanations
    ]
    logger.info("gate: investigated, flagged (%.0fms)", (time.monotonic() - start) * 1000)
    return AnalyzeResponse(is_clean=False, cards=cards)


# Serve the built frontend from the same origin as the API, when one is
# present. Only the deployed image has this; local development runs Vite on
# :5173 against uvicorn on :8000 and this directory doesn't exist, so nothing
# changes there.
#
# Mounted LAST on purpose: Starlette matches routes in registration order, so
# /health and /analyze above are found before this catch-all mount. Registering
# it earlier would swallow them and serve index.html for the API.
if config.STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    # html=True serves index.html for "/" and falls back to it for unknown
    # paths, which is what a single-page app needs.
    app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")
    logger.info("serving frontend from %s", config.STATIC_DIR)
