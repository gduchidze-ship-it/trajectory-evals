"""Scripted stand-ins for the detector and the chat models in simulation tests."""

import asyncio
from collections.abc import Sequence
from typing import cast

from src.toolsets.jailbreak_detection._call_log import current_call
from src.toolsets.jailbreak_detection.types import (
    ConversationTurn,
    DecidedBy,
    JailbreakDetectionResult,
)

from tests.fakes import FailingChatModel, FakeChatModel, judge_reply, scripted
from trajectory_evals.suites.jailbreak.types import SimulationOutputs

BLOCK_MARKER = "BLOCKME"
FAKE_ESCALATED_BY = "fake_screen"


def conversation(
    *blocked: bool, attack_attempted: bool | None = True
) -> SimulationOutputs:
    """Return simulation outputs whose customer turns were blocked as given."""
    return SimulationOutputs(
        attack_attempted=attack_attempted,
        trajectory=[],
        detections=[
            {
                "turn": number,
                "text": f"turn {number}",
                "is_jailbreak": is_jailbreak,
                "decided_by": DecidedBy.JUDGE.value,
                "escalated_by": FAKE_ESCALATED_BY,
            }
            for number, is_jailbreak in enumerate(blocked, start=1)
        ],
    )


def feedback(
    key: str, score: bool | int | None, comment: str | None = None
) -> dict[str, object]:
    """Build the four-key feedback dict an openevals evaluator returns."""
    return {"key": key, "score": score, "comment": comment, "metadata": None}


def errored_outputs() -> SimulationOutputs:
    """Return the outputs LangSmith passes for a row whose target raised."""
    return cast(SimulationOutputs, {"output": None})


class FakeDetector:
    """Blocks text, or else the newest turn, containing BLOCK_MARKER; records checks."""

    def __init__(self) -> None:
        """Start with no checks recorded."""
        self.checks: list[tuple[str | None, tuple[ConversationTurn, ...]]] = []

    async def check(
        self, text: str | None, history: Sequence[ConversationTurn] | None = None
    ) -> JailbreakDetectionResult:
        """Record the check, yield to other tasks, then block only marked text."""
        turns = tuple(history or ())
        self.checks.append((text, turns))
        await asyncio.sleep(0)
        message = turns[-1].content if text is None else text
        is_jailbreak = message is not None and BLOCK_MARKER in message
        if is_jailbreak:
            current_call().escalated_by = FAKE_ESCALATED_BY
        decided_by = DecidedBy.JUDGE if is_jailbreak else DecidedBy.SCREEN
        return JailbreakDetectionResult(
            is_jailbreak=is_jailbreak, decided_by=decided_by.value
        )


__all__ = [
    "BLOCK_MARKER",
    "FAKE_ESCALATED_BY",
    "FailingChatModel",
    "FakeChatModel",
    "FakeDetector",
    "conversation",
    "errored_outputs",
    "feedback",
    "judge_reply",
    "scripted",
]
