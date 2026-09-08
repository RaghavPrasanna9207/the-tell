"""FastAPI app. Wires /analyze to Layer 1 (classify) + Layer 2 (explain).

Layer 1 runs as a two-stage cascade rather than calling the teacher
directly: the distilled ModernBERT student (app.student) screens every
message first as a deliberately high-recall gate, and the (slow, ~5s)
teacher is only woken up when the student finds something worth
investigating. See docs/DESIGN_RULES.md and the plan's Phase A1 — this is what makes
the distilled student a real part of the running system rather than a
benchmark-only artifact. If the student model isn't present (e.g.
eval/distill.py has never been run), the gate is skipped entirely and every
message goes straight to the teacher, so a fresh clone still works.

If Ollama isn't running or the model isn't pulled, /analyze returns a 503
with an actionable message — it never falls back to fabricated output or a
cloud API. See docs/DESIGN_RULES.md.
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
