"""Scripted stand-ins for chat models and agents that any test can use."""

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from trajectory_evals.core.types import AgentTurn, ChatCompletionMessage

JUDGE_REASONING = "scripted reasoning"
FAKE_CONTEXT_PREFIX = "fake-context-"


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


@dataclass(frozen=True)
class AgentCall:
    """One send a FakeAgent received, with copies of what it was sent."""

    text: str
    context_id: str | None
    history: tuple[ChatCompletionMessage, ...]
    prompt_overrides: Mapping[str, str] | None
    context: Mapping[str, Any] | None


type Reply = str | AgentTurn


class FakeAgent:
    """An Agent replying from a script or a function of each call; records calls."""

    def __init__(
        self, *replies: Reply, respond: Callable[[AgentCall], Reply] | None = None
    ) -> None:
        """Reply with replies in order, or with respond(call) when it is given."""
        if replies and respond is not None:
            raise TypeError("pass scripted replies or respond, not both")
        self.calls: list[AgentCall] = []
        self._replies = deque(replies)
        self._respond = respond
        self._conversations = 0

    async def send(
        self,
        text: str,
        *,
        context_id: str | None = None,
        history: Sequence[ChatCompletionMessage] = (),
        prompt_overrides: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> AgentTurn:
        """Record the call and return the next reply, a str as one assistant message."""
        overrides = None if prompt_overrides is None else dict(prompt_overrides)
        call = AgentCall(
            text=text,
            context_id=context_id,
            history=tuple(history),
            prompt_overrides=overrides,
            context=None if context is None else dict(context),
        )
        self.calls.append(call)
        reply = self._next_reply(call)
        if isinstance(reply, AgentTurn):
            return reply
        message: ChatCompletionMessage = {"role": "assistant", "content": reply}
        context_id = context_id or self._new_context()
        return AgentTurn(text=reply, messages=(message,), context_id=context_id)

    def _next_reply(self, call: AgentCall) -> Reply:
        """Return respond(call), or the next scripted reply."""
        if self._respond is not None:
            return self._respond(call)
        if not self._replies:
            raise AssertionError(f"FakeAgent has no scripted reply for {call.text!r}")
        return self._replies.popleft()

    def _new_context(self) -> str:
        """Return the next conversation id, numbered from 1 in order of creation."""
        self._conversations += 1
        return f"{FAKE_CONTEXT_PREFIX}{self._conversations}"
