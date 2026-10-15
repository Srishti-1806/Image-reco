"""Block 7: decision logic, kept pure so it is trivially unit-testable."""
from dataclasses import dataclass


@dataclass
class Decision:
    status: str          # ALLOWED | DENIED
    action: str          # ACCESS_GRANTED | ALERT | NONE
    reason: str


def decide(score: float | None, threshold: float, zone: str) -> Decision:
    if score is None:
        return Decision("DENIED", "ALERT" if zone == "secure" else "NONE", "no_enrolled_identities")
    if score >= threshold:
        return Decision("ALLOWED", "ACCESS_GRANTED", "match")
    return Decision("DENIED", "ALERT" if zone == "secure" else "NONE", "below_threshold")
