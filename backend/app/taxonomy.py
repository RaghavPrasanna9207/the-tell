"""Single source of truth for the manipulation-technique taxonomy.

Every other part of the system imports from here: the JSON schema fed to
Ollama's constrained decoder (Layer 1), the FastAPI response model, the eval
label space, and the key set validated against `data/counter_moves.yaml`.
Do not redefine the technique list anywhere else.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Technique(str, Enum):
    """The 11 manipulation techniques, derived from documented digital-arrest
    and UPI scam scripts (I4C advisories, public reporting) rather than
    invented. See `data/counter_moves.yaml` for the grounded fact behind each.
    """

    MANUFACTURED_URGENCY = "manufactured_urgency"
    """Artificial deadline — "within 2 hours", "account closes today"."""

    FALSE_AUTHORITY = "false_authority"
    """Impersonating Police / CBI / ED / RBI / TRAI / a bank."""

    ISOLATION = "isolation"
    """"Don't tell your family", "sub-judice", "stay on this call"."""

    FEAR_OF_CONSEQUENCE = "fear_of_consequence"
    """Arrest, jail, named legal sections (420, NDPS), account freeze."""

    VERIFICATION_THEATER = "verification_theater"
    """Fake FIR, fake ID card, fake court order, "verify funds in RBI escrow"."""

    CHANNEL_SWITCH = "channel_switch"
    """"Call this number", "download AnyDesk/TeamViewer", move to WhatsApp/Telegram."""

    PAYMENT_IRREVERSIBILITY = "payment_irreversibility"
    """UPI *collect* request, scan-QR-to-*receive* trick, gift cards, crypto."""

    TRUST_TRANSFER = "trust_transfer"
    """Impersonating a known person or brand — "Mom, I lost my phone"."""

    RECIPROCITY_HOOK = "reciprocity_hook"
    """Unsolicited gift, prize, "refund owed to you"."""

    FAKE_SCARCITY = "fake_scarcity"
    """"Only 3 slots left", "offer expires"."""

    SUNK_COST_PRESSURE = "sunk_cost_pressure"
    """"You've already paid ₹X — just one more step"."""


ALL_TECHNIQUES: tuple[Technique, ...] = tuple(Technique)


# Enum member docstrings above are documentation only — Python does not
# attach them at runtime, so anything that needs the description at runtime
# (prompt-building for Layer 1, the frontend's "what does this mean" copy)
# must read from this mapping instead. Keep the two in sync.
TECHNIQUE_DESCRIPTIONS: dict[Technique, str] = {
    Technique.MANUFACTURED_URGENCY: (
        "An artificial deadline meant to stop the reader from pausing to think — "
        "e.g. 'within 2 hours', 'account closes today'."
    ),
    Technique.FALSE_AUTHORITY: (
        "Impersonating a real institution — police, CBI, ED, RBI, TRAI, or a bank — "
        "to borrow its credibility."
    ),
    Technique.ISOLATION: (
        "An instruction to keep the interaction secret from family or friends — "
        "e.g. 'don't tell your family', 'this is sub-judice', 'stay on this call'."
    ),
    Technique.FEAR_OF_CONSEQUENCE: (
        "A threat of arrest, jail, or account freeze, often citing a specific legal "
        "section (like IPC 420 or the NDPS Act) to sound credible."
    ),
    Technique.VERIFICATION_THEATER: (
        "Fake proof — a forged FIR, ID card, or court order, or a request to 'verify' "
        "funds by sending them to an escrow account."
    ),
    Technique.CHANNEL_SWITCH: (
        "An instruction to move to a different channel or install remote-access "
        "software — e.g. 'call this number', 'download AnyDesk', switch to WhatsApp."
    ),
    Technique.PAYMENT_IRREVERSIBILITY: (
        "A payment framed as a receipt — a UPI 'collect' request or a 'scan to receive' "
        "trick, gift cards, or crypto, where completing the action actually sends money."
    ),
    Technique.TRUST_TRANSFER: (
        "Impersonating a specific person or brand the reader already trusts — "
        "e.g. 'Mom, I lost my phone, this is my new number'."
    ),
    Technique.RECIPROCITY_HOOK: (
        "An unsolicited gift, prize, or refund offer designed to create a felt "
        "obligation to respond."
    ),
    Technique.FAKE_SCARCITY: (
        "An artificial limit meant to rush the decision — "
        "e.g. 'only 3 slots left', 'offer expires today'."
    ),
    Technique.SUNK_COST_PRESSURE: (
        "Leveraging money already paid to justify one more payment — "
        "e.g. 'you've already paid ₹X, just one more step'."
    ),
}


class Detection(BaseModel):
    """One manipulation technique found in a message, grounded to the exact
    text that triggered it.
    """

    technique: Technique
    span: str = Field(
        description=(
            "The exact substring from the original message that demonstrates "
            "this technique. Must be copied verbatim, not paraphrased."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0)


class Analysis(BaseModel):
    """Layer 1 output: every manipulation technique detected in a message.

    An empty `detections` list means no technique was found — this is
    "no manipulation techniques detected," never "this message is safe."
    See CLAUDE.md.
    """

    detections: list[Detection] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.detections) == 0

    @property
    def technique_ids(self) -> set[Technique]:
        return {d.technique for d in self.detections}


def analysis_json_schema() -> dict:
    """JSON schema for `Analysis`, passed to Ollama's `format` parameter to
    constrain decoding at the token level. Malformed output becomes
    impossible rather than unlikely — no retry loop, no `json.loads` in a
    `try`.
    """
    return Analysis.model_json_schema()
