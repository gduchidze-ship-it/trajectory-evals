"""The openevals app: each simulated user message goes to the agent under test."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from trajectory_evals.core.protocols import Agent
from trajectory_evals.core.types import AgentOutputs, ChatCompletionMessage
from trajectory_evals.simulation.settings import settings


@dataclass
class _Thread:
    """One conversation: the agent's context id, its transcript and how it stands."""

    context_id: str | None = None
    messages: list[ChatCompletionMessage] = field(default_factory=list)
    reply: str = ""
    trajectory_reported: bool = True
    ended: bool = False


class A2AConversationApp:
    """Sends each user message to the agent and keeps every thread's full transcript.

    openevals sees only each turn's final reply; the transcript also keeps the tool
    calls and results the agent reported. A message holding the stop token ends the
    thread without reaching the agent or the transcript.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        prompt_overrides: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
        stop_token: str,
    ) -> None:
        """Hold the agent and what every message carries; no thread has started."""
        self._agent = agent
        self._prompt_overrides = prompt_overrides or None
        self._context = context
        self._stop_token = stop_token
        self._threads: defaultdict[str, _Thread] = defaultdict(_Thread)

    async def respond(
        self, message: ChatCompletionMessage, *, thread_id: str, **_: Any
    ) -> ChatCompletionMessage:
        """Return the agent's final reply to the message, continuing the thread.

        A message holding the stop token gets a placeholder reply instead and never
        reaches the agent. An AgentError from the agent propagates.
        """
        thread = self._threads[thread_id]
        text = _text_of(message)
        if self._stop_token in text:
            thread.ended = True
            return {"role": "assistant", "content": settings.prompts.stopped_reply}
        turn = await self._agent.send(
            text,
            context_id=thread.context_id,
            prompt_overrides=self._prompt_overrides,
            context=self._context,
        )
        thread.context_id = turn.context_id
        thread.messages += [{"role": "user", "content": text}, *turn.messages]
        thread.reply = turn.text
        thread.trajectory_reported &= turn.trajectory_reported
        return {"role": "assistant", "content": turn.text}

    def transcript(self, thread_id: str) -> list[ChatCompletionMessage]:
        """Return the thread's user messages and agent messages so far, in order."""
        return list(self._thread(thread_id).messages)

    def ended(self, thread_id: str) -> bool:
        """Return whether the simulated user sent the stop token on the thread."""
        return self._thread(thread_id).ended

    def outputs(self, thread_id: str) -> AgentOutputs:
        """Return the thread's last agent reply and its full transcript."""
        thread = self._thread(thread_id)
        return AgentOutputs(
            text=thread.reply,
            messages=list(thread.messages),
            trajectory_reported=thread.trajectory_reported,
        )

    def _thread(self, thread_id: str) -> _Thread:
        """Return the thread, or a fresh one without recording it if it never began."""
        return self._threads.get(thread_id) or _Thread()


def _text_of(message: ChatCompletionMessage) -> str:
    """Return the message's text, joining its text blocks when content is a list."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


__all__ = ["A2AConversationApp"]
