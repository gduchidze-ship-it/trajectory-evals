"""Scripted stand-ins for the detector and the chat models in simulation tests."""

import asyncio
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from trajectory_evals.suites.jailbreak.types import SimulationOutputs
from src.toolsets.jailbreak_detection._call_log import current_call
from src.toolsets.jailbreak_detection.types import (
    ConversationTurn,
    DecidedBy,
    JailbreakDetectionResult,
)

BLOCK_MARKER = "BLOCKME"
FAKE_ESCALATED_BY = "fake_screen"
JUDGE_REASONING = "scripted reasoning"


class FakeChatModel(GenericFakeChatModel):
    """Replies with scripted messages, records each prompt and accepts any tools."""

    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Record the prompt, then return the next scripted reply."""
        self.prompts.append(messages)
        return super()._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Ignore the tools, so structured output parses the scripted tool call."""
        return self


class FailingChatModel(FakeChatModel):
    """A chat model whose every call fails, like an unreachable judge."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Record the prompt, then fail as an unreachable API would."""
        self.prompts.append(messages)
        raise RuntimeError("judge unavailable")


def scripted(*replies: str | AIMessage) -> FakeChatModel:
    """Return a FakeChatModel that gives replies in order."""
    return FakeChatModel(messages=iter(replies))


def judge_reply(score: bool) -> AIMessage:
    """Return the tool call that an openevals judge parses as its score."""
    arguments = {"reasoning": JUDGE_REASONING, "score": score}
    return AIMessage(
        content="", tool_calls=[{"name": "score", "args": arguments, "id": "call-1"}]
    )


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
