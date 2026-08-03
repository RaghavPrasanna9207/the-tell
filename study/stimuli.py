"""The six fixed stimuli for the pre-registered study (see PREREGISTRATION.md
"Stimuli"). Same six messages, same order, shown to every participant
regardless of arm — only the flag/explanation shown alongside each one
differs by arm.

Messages 1-4 are scam stimuli (primary analysis). Messages 5-6 are
legitimate messages testing the false-alarm rate (secondary analysis, H2).
None of these are pulled from the training/eval corpus — they're written
fresh for the study so the classifier sees them cold, same as a real
participant would.
"""

STIMULI = [
    {
        "id": "S1_digital_arrest",
        "kind": "scam",
        "text": (
            "This is Sub-Inspector Vikram Rathore, Mumbai Cyber Crime Cell. Your Aadhaar "
            "number is linked to a parcel containing illegal substances seized at Mumbai "
            "airport. An FIR under NDPS Act and IPC 420 has been filed against you. You must "
            "stay on this video call and not disconnect until verification is complete. Do "
            "not inform any family member — this matter is sub-judice. Failure to cooperate "
            "within 30 minutes will result in a non-bailable warrant and arrest."
        ),
    },
    {
        "id": "S2_upi_collect",
        "kind": "scam",
        "text": (
            "Congratulations! A refund of Rs 4,999 from your recent Amazon order has been "
            "approved. To receive it instantly, open your UPI app and accept the payment "
            "request we are sending now, then enter your UPI PIN to confirm receipt. This "
            "refund offer expires in 15 minutes."
        ),
    },
    {
        "id": "S3_family_impersonation",
        "kind": "scam",
        "text": (
            "Hi Mom, I lost my phone and this is a friend's number I'm using for now. I'm "
            "stuck at the bank and short on cash for an urgent payment before it closes — "
            "can you send Rs 15,000 to this account right away? I'll explain everything "
            "when I see you, please don't call Dad about it, he'll just worry."
        ),
    },
    {
        "id": "S4_lottery_prize",
        "kind": "scam",
        "text": (
            "GOOD NEWS! Your mobile number has won Rs 25,00,000 in the KBC Lucky Draw 2026, "
            "sponsored by Reliance Jio. Only 3 winners have been selected today. To claim "
            "your prize, contact our claims officer on WhatsApp within the next 2 hours and "
            "share your bank details for the prize transfer."
        ),
    },
    {
        "id": "S5_bank_otp",
        "kind": "legitimate",
        "text": (
            "Your OTP for the transaction of Rs 2,340 at Big Bazaar is 483920. Valid for 10 "
            "minutes. Do not share this OTP with anyone. - HDFC Bank"
        ),
    },
    {
        "id": "S6_delivery_notification",
        "kind": "legitimate",
        "text": (
            "Your Swiggy order #48213 from Sagar Ratna has been picked up and is on its way. "
            "Estimated delivery by 8:45 PM. Track your order in the app."
        ),
    },
]

STIMULI_BY_ID = {s["id"]: s for s in STIMULI}
