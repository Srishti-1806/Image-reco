"""Fail-closed adapter for the InsightFace presentation-attack detection verdict."""
from abc import ABC, abstractmethod


class LivenessChecker(ABC):
    @abstractmethod
    def is_live(self, frame, face) -> bool: ...


class InsightFaceLiveness(LivenessChecker):
    def is_live(self, frame, face) -> bool:
        result = getattr(face, "liveness", None)
        return (
            isinstance(result, dict)
            and result.get("status") == "ok"
            and result.get("is_live") is True
        )


def build() -> LivenessChecker:
    return InsightFaceLiveness()
