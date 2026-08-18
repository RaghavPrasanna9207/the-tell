"""Layer 1 pre-filter: the distilled ModernBERT student (see
eval/distill.py), run as a deliberately high-recall gate in front of the
Ollama teacher — see CLAUDE.md and the plan's Phase A1 ("a cascade, not a
swap"). The student has no notion of verbatim spans, so it can never
answer /analyze on its own; its only job is deciding whether the teacher
(the slower but span-accurate stage that Layer 2 depends on) is worth
waking up for this message. Model is loaded once and cached at first use,
not per request.
"""

from pathlib import Path

from app.taxonomy import Technique

STUDENT_MODEL_DIR = Path(__file__).parent.parent / "models" / "student"

# Deliberately far below any per-technique F1-optimal threshold
# (eval/metrics.py's THRESHOLD_GRID bottoms out at 0.05, and several
# techniques already pick that floor as their own optimum) — see
# eval/gate_calibration.py for the measured recall/wake-rate this achieves
# against the real gold set. A gate firing unnecessarily just costs ~5s of
# teacher latency; a gate that stays silent means Layer 2 never runs for
# that message at all, which is the failure mode this threshold is chosen
# to avoid.
GATE_THRESHOLD = 0.05

_TECHNIQUE_NAMES = [t.value for t in Technique]

_model = None
_tokenizer = None


def is_available() -> bool:
    """False on a fresh clone before eval/distill.py has ever been run —
    callers must treat that as "skip the gate, always call the teacher",
    not an error. See main.py."""
    return STUDENT_MODEL_DIR.exists()


def warm_up() -> None:
    """Eagerly load the model (first-time torch/transformers import +
    weight load + GPU transfer) so the cost is paid once at server startup
    instead of stalling whichever user happens to send the first request —
    measured at ~20-30s cold on this machine. See main.py's startup hook."""
    if is_available():
        _load()


def _load() -> None:
    global _model, _tokenizer
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(str(STUDENT_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(STUDENT_MODEL_DIR))
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    _model = model


def n_params() -> int:
    _load()
    return sum(p.numel() for p in _model.parameters())


def predict_proba_array(text: str):
    """Raw sigmoid probabilities, one per technique, in `app.taxonomy.Technique`
    order. Building block for predict_proba() and for eval scripts that want
    the array form directly (e.g. eval/compare.py, eval/gate_calibration.py)
    — the single place the student model is loaded and run, so the live
    gate and the eval scripts are guaranteed to exercise the same inference
    path."""
    import torch

    _load()
    device = next(_model.parameters()).device
    inputs = _tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    return torch.sigmoid(logits).cpu().numpy()[0]


def predict_proba(text: str) -> dict[str, float]:
    return dict(zip(_TECHNIQUE_NAMES, predict_proba_array(text).tolist()))


def should_investigate(text: str, threshold: float = GATE_THRESHOLD) -> bool:
    """True if any technique's student probability clears the gate
    threshold — i.e. whether classify() (the teacher) is worth calling for
    this message. Deliberately asymmetric: a false positive here only
    costs latency (the teacher gets called and likely also finds nothing);
    a false negative means Layer 2 never runs for a message that needed
    it, which is the more expensive mistake."""
    return bool((predict_proba_array(text) >= threshold).any())
