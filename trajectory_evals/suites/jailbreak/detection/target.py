"""The aevaluate target that screens one detection case with a Detector."""

from trajectory_evals.suites.jailbreak.protocols import Detector
from trajectory_evals.suites.jailbreak.types import DetectionInputs, DetectionOutputs
from trajectory_evals.suites.jailbreak.utils import _detector_arguments, _has_text
from src.toolsets.jailbreak_detection._call_log import track_call
from src.toolsets.jailbreak_detection.types import ConversationTurn


class DetectionTarget:
    """Screens each case with a Detector and reports the verdict and its route."""

    def __init__(self, detector: Detector) -> None:
        """Hold the detector that screens every case."""
        self._detector = detector

    async def __call__(self, inputs: DetectionInputs) -> DetectionOutputs:
        """Check the case as prod calls the detector, in a tracked call, and log it."""
        text, prior = _message_and_prior(inputs)
        arguments = _detector_arguments(inputs["channel"], text, prior)
        with track_call(None, None) as call:
            result = await self._detector.check(*arguments)
        return DetectionOutputs(
            is_jailbreak=result.is_jailbreak,
            decided_by=result.decided_by,
            escalated_by=call.escalated_by,
            history_read=call.history_read,
            judge_reminded=call.judge_reminded,
        )


def _message_and_prior(
    inputs: DetectionInputs,
) -> tuple[str | None, list[ConversationTurn]]:
    """Return the message and the turns before it; without text, the newest turn."""
    prior = [ConversationTurn.model_validate(turn) for turn in inputs["history"]]
    text = inputs["text"]
    if _has_text(text):
        return text, prior
    *prior, current = prior
    return current.content, prior
