"""Layer 1 pre-filter: the distilled ModernBERT student (see
eval/distill.py), run as a deliberately high-recall gate in front of the
Ollama teacher — see docs/DESIGN_RULES.md and the plan's Phase A1 ("a cascade, not a
swap"). The student has no notion of verbatim spans, so it can never
answer /analyze on its own; its only job is deciding whether the teacher
(the slower but span-accurate stage that Layer 2 depends on) is worth
waking up for this message. Model is loaded once and cached at first use,
not per request.
"""

from app import config
from app.taxonomy import Technique

# Re-exported from app.config (which reads the environment, defaulting to
# exactly these previous hardcoded values) so every existing import site —
# eval/gate_calibration.py, eval/distill.py, eval/cascade_eval.py, the tests
# — keeps working unchanged.
STUDENT_MODEL_DIR = config.STUDENT_MODEL_DIR

# Deliberately far below any per-technique F1-optimal threshold
# (eval/metrics.py's THRESHOLD_GRID bottoms out at 0.05, and several
# techniques already pick that floor as their own optimum) — see
# eval/gate_calibration.py for the measured recall/wake-rate this achieves
# against the real gold set. A gate firing unnecessarily just costs one
# teacher call; a gate that stays silent means Layer 2 never runs for that
# message at all, which is the failure mode this threshold is chosen to
# avoid. eval/cascade_eval.py measures what that costs end to end.
GATE_THRESHOLD = config.GATE_THRESHOLD

_TECHNIQUE_NAMES = [t.value for t in Technique]

_model = None
_tokenizer = None


def model_source() -> str | None:
    """Where the student weights come from, or None if there are none.

    A local training output wins; otherwise a Hugging Face repo id if one is
    configured (the deployed image has no local training output, so it pulls
    the published student instead). None means no gate — see is_available."""
    if STUDENT_MODEL_DIR.exists():
        return str(STUDENT_MODEL_DIR)
    return config.STUDENT_HF_REPO or None


def is_available() -> bool:
    """False on a fresh clone before eval/distill.py has ever been run and
    with no STUDENT_HF_REPO configured — callers must treat that as "skip the
    gate, always call the teacher", not an error. See main.py."""
    return model_source() is not None


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

    # from_pretrained takes a local path or a Hub repo id in the same
    # argument, so the deployed image (no local training output, pulls the
    # published student) and a dev machine (local output) share one path.
    source = model_source()
    if source is None:
        raise RuntimeError("no student model available — callers must check is_available() first")
    _tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
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
