"""A scripted detector, transcripts and prod rows the jailbreak example tests share."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from src.toolsets.jailbreak_detection._call_log import CallLog, current_call
from src.toolsets.jailbreak_detection.types import (
    ConversationTurn,
    DecidedBy,
    JailbreakDetectionResult,
)

from examples.jailbreak.types import DetectionResult
from trajectory_evals.core.agent_config import AgentConfig, load_agent
from trajectory_evals.core.settings import REPO_ROOT
from trajectory_evals.core.types import ChatCompletionMessage

EXAMPLE_DIR = REPO_ROOT / "examples" / "jailbreak"
BLOCK = "BLOCKME"  # the judge blocks it after a reminder; the guardrail escalated
ESCALATE = "ESCALATE"  # the judge allows it; the screen read history, escalated
FAIL_OPEN = "FAILOPEN"  # the fail policy allows it; the screen escalated it
CRASH = "CRASH"  # the detector raises
_ROUTES: dict[str, tuple[bool, DecidedBy, CallLog]] = {
    BLOCK: (
        True,
        DecidedBy.JUDGE,
        CallLog(escalated_by="guardrail", judge_reminded=True),
    ),
    ESCALATE: (
        False,
        DecidedBy.JUDGE,
        CallLog(escalated_by="screen", history_read=True),
    ),
    FAIL_OPEN: (False, DecidedBy.FAIL_POLICY, CallLog(escalated_by="screen")),
}
_QUIET: tuple[bool, DecidedBy, CallLog] = (False, DecidedBy.SCREEN, CallLog())


class FakeDetector:
    """Decides by the marker in the text, else the newest turn; records each check."""

    def __init__(self) -> None:
        """Start with no checks recorded."""
        self.checks: list[tuple[str | None, tuple[ConversationTurn, ...]]] = []

    async def check(
        self, text: str | None, history: Sequence[ConversationTurn] | None = None
    ) -> JailbreakDetectionResult:
        """Record the check, log the route from a child task, return the verdict."""
        turns = tuple(history or ())
        self.checks.append((text, turns))
        message = str(turns[-1].content if text is None else text)
        if CRASH in message:
            raise RuntimeError("detector crashed")
        is_jailbreak, decided_by, route = next(
            (route for marker, route in _ROUTES.items() if marker in message), _QUIET
        )
        await asyncio.create_task(_record_route(route))
        return JailbreakDetectionResult(
            is_jailbreak=is_jailbreak, decided_by=decided_by.value
        )


async def _record_route(route: CallLog) -> None:
    """Write the route into the running call's log, as a real detector does."""
    call = current_call()
    call.escalated_by = route.escalated_by
    call.history_read = route.history_read
    call.judge_reminded = route.judge_reminded


def detection(
    is_jailbreak: bool = False, decided_by: str = "screen", escalated_by: str = "none"
) -> DetectionResult:
    """Return a detect_jailbreak result."""
    return DetectionResult(
        is_jailbreak=is_jailbreak,
        decided_by=decided_by,
        escalated_by=escalated_by,
        failed_open=decided_by == "fail_policy",
        history_read=False,
        judge_reminded=False,
    )


def turn(
    text: str, result: DetectionResult, number: int = 1
) -> list[ChatCompletionMessage]:
    """Return one user turn: the message, the tool call and result, then a reply."""
    call_id = f"call_{number}"
    function = {"name": "detect_jailbreak", "arguments": json.dumps({"text": text})}
    call = {"id": call_id, "type": "function", "function": function}
    tool_result = {
        "role": "tool",
        "content": json.dumps(result),
        "tool_call_id": call_id,
    }
    return [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        cast(ChatCompletionMessage, tool_result),
        {"role": "assistant", "content": "reply"},
    ]


def detection_outputs(is_jailbreak: bool, decided_by: str) -> dict[str, Any]:
    """Return a golden target's outputs for one screened message."""
    messages = turn("Hello", detection(is_jailbreak, decided_by))
    return {"text": "reply", "messages": messages}


def conversation_outputs(*blocked: bool, kind: str = "attack") -> dict[str, Any]:
    """Return a simulation target's outputs whose user turns were blocked as given."""
    messages = [
        message
        for number, is_blocked in enumerate(blocked, start=1)
        for message in turn(f"turn {number}", detection(is_blocked), number)
    ]
    return {"text": "reply", "messages": messages, "kind": kind}


def errored_outputs() -> dict[str, Any]:
    """Return the outputs LangSmith passes for a row whose target raised."""
    return {"output": None}


def feedback(
    key: str, score: bool | int | None, comment: str | None = None
) -> dict[str, object]:
    """Build the four-key feedback dict an openevals evaluator returns."""
    return {"key": key, "score": score, "comment": comment, "metadata": None}


def write_jsonl(path: Path, rows: Sequence[dict[str, Any] | str]) -> Path:
    """Write rows to path as JSON lines, strings verbatim, and return the path."""
    lines = (row if isinstance(row, str) else json.dumps(row) for row in rows)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def example_agent(name: str) -> AgentConfig:
    """Load the example's detector or assistant agent folder."""
    return load_agent(EXAMPLE_DIR / name)


__all__ = [
    "BLOCK",
    "CRASH",
    "ESCALATE",
    "EXAMPLE_DIR",
    "FAIL_OPEN",
    "FakeDetector",
    "conversation_outputs",
    "detection",
    "detection_outputs",
    "errored_outputs",
    "example_agent",
    "feedback",
    "turn",
    "write_jsonl",
]
