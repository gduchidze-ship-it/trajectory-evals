"""Checks that the real detection service fits the Detector protocol."""

import asyncio
from collections.abc import Mapping

from trajectory_evals.suites.jailbreak.protocols import Detector
from src.toolsets.jailbreak_detection.service import JailbreakDetectionService
from src.toolsets.jailbreak_detection.settings import settings
from src.toolsets.jailbreak_detection.types import Finding, InboundMessage, Verdict


async def _unreachable_judge(
    message: InboundMessage, findings: Mapping[str, Finding]
) -> Verdict | None:
    """Fail the test if the service escalates a message to the judge."""
    raise AssertionError(f"judge called for {message.text!r}")


def test_detection_service_is_a_detector() -> None:
    """A service typed as a Detector screens a message through check."""
    detector: Detector = JailbreakDetectionService(
        detectors={}, judge=_unreachable_judge, settings=settings
    )
    result = asyncio.run(detector.check("When does the showroom open?"))
    assert (result.is_jailbreak, result.decided_by) == (False, "screen")
