"""FastAPI app. Wires /analyze to Layer 1 (classify) + Layer 2 (explain).

If Ollama isn't running or the model isn't pulled, /analyze returns a 503
with an actionable message — it never falls back to fabricated output or a
cloud API. See CLAUDE.md.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.classify import classify
from app.explain import explain_all
from app.llm import OllamaUnavailableError
from app.taxonomy import Technique

app = FastAPI(title="ScamShield Explainer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
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
    try:
        classify_result = classify(req.text)
        if classify_result.analysis.is_clean:
            return AnalyzeResponse(is_clean=True, cards=[])

        explanations = explain_all(classify_result.analysis.detections)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    cards = [
        TechniqueCard(
            technique=result.detection.technique,
            plain_name=result.counter_move.plain_name,
            span=result.detection.span,
            confidence=result.detection.confidence,
            explanation=result.explanation,
            source=result.counter_move.source,
        )
        for result in explanations
    ]
    return AnalyzeResponse(is_clean=False, cards=cards)
