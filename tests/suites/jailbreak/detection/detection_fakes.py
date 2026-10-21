"""Target outputs and expected feedback shared by the detection evaluator tests."""

from typing import cast

from trajectory_evals.suites.jailbreak.types import DetectionOutputs, DetectionReference


def detection_outputs(is_jailbreak: bool, decided_by: str) -> DetectionOutputs:
    """Build a target output with a verdict and the cascade step that decided it."""
    return DetectionOutputs(
        is_jailbreak=is_jailbreak,
        decided_by=decided_by,
        escalated_by="none" if decided_by == "screen" else "screen",
        history_read=False,
        judge_reminded=False,
    )


def errored_outputs() -> DetectionOutputs:
    """Return the outputs LangSmith gives evaluators when the target raised."""
    return cast(DetectionOutputs, {"output": None})


def empty_reference() -> DetectionReference:
    """Return the reference outputs LangSmith gives for an example with no label."""
    return cast(DetectionReference, {})


def unknown_label_reference() -> DetectionReference:
    """Return a reference whose label is outside ExpectedAction."""
    return cast(DetectionReference, {"expected_action": "UNSURE"})


def feedback(
    key: str, score: bool | None, comment: str | None = None
) -> dict[str, object]:
    """Build the four-key feedback dict an openevals evaluator returns."""
    return {"key": key, "score": score, "comment": comment, "metadata": None}
