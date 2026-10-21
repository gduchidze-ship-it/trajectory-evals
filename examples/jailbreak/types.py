"""Typed shapes the jailbreak responder, graders and converter share."""

from typing import Literal, TypedDict

ExpectedAction = Literal["ALLOWED", "BLOCKED", "REVIEW"]


class DetectionResult(TypedDict):
    """The detect_jailbreak tool result the responder reports on every turn.

    escalated_by names the detector that escalated first, or "none", as the
    detector's CallLog recorded it; failed_open is decided_by == "fail_policy".
    """

    is_jailbreak: bool
    decided_by: str
    escalated_by: str
    failed_open: bool


__all__ = ["DetectionResult", "ExpectedAction"]
