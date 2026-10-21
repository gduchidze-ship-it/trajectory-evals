"""Structural interfaces that the jailbreak evaluations take as collaborators."""

from collections.abc import Sequence
from typing import Protocol

from src.toolsets.jailbreak_detection.types import (
    ConversationTurn,
    JailbreakDetectionResult,
)

from trajectory_evals.core.protocols import CaseSource, DatasetSink


class Detector(Protocol):
    """Screens one inbound message the way detect_jailbreak does."""

    async def check(
        self, text: str | None, history: Sequence[ConversationTurn] | None = None
    ) -> JailbreakDetectionResult:
        """Return the verdict on text, or on the newest customer turn of history."""


__all__ = ["CaseSource", "DatasetSink", "Detector"]
