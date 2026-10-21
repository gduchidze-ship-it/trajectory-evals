"""The binary judge of whether an agent's latest turn violates a red-team policy."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from openevals.llm import create_async_llm_as_judge
from openevals.utils import _chat_completion_messages_to_string

from trajectory_evals.core.types import ChatCompletionMessage
from trajectory_evals.simulation.settings import settings


class AttackJudge:
    """Grades only the agent's latest turn of a transcript against the policy."""

    def __init__(self, model: BaseChatModel, policy: str) -> None:
        """Build the judge over model; policy is the text the agent must keep to."""
        self._judge: Callable[..., Awaitable[Any]] = create_async_llm_as_judge(
            prompt=settings.prompts.attack_judge,
            judge=model,
            feedback_key=settings.keys.attack_succeeded,
        )
        self._policy = policy

    async def __call__(self, transcript: Sequence[ChatCompletionMessage]) -> bool:
        """Return whether the turn after the last user message violates the policy.

        The judge sees what came before as context only. Raises whatever the judge
        model raises.
        """
        start = _latest_turn_start(transcript)
        result = await self._judge(
            policy=self._policy,
            conversation=_chat_completion_messages_to_string(list(transcript[:start])),
            reply=_chat_completion_messages_to_string(list(transcript[start:])),
        )
        return bool(result["score"])


def _latest_turn_start(transcript: Sequence[ChatCompletionMessage]) -> int:
    """Return the index just after the last user message, or 0 if there is none."""
    users = [
        index for index, message in enumerate(transcript) if message["role"] == "user"
    ]
    return users[-1] + 1 if users else 0


__all__ = ["AttackJudge"]
