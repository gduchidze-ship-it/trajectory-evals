"""The aevaluate target that sends one golden case's conversation to an Agent."""

from collections.abc import Mapping

from trajectory_evals.core.protocols import Agent
from trajectory_evals.core.types import AgentInputs, AgentOutputs
from trajectory_evals.golden.settings import settings


class GoldenTarget:
    """Sends each case's last user message, with the messages before it as history."""

    def __init__(
        self, agent: Agent, *, prompt_overrides: Mapping[str, str] | None = None
    ) -> None:
        """Hold the agent, and the prompt overrides sent with every case."""
        self._agent = agent
        self._prompt_overrides = prompt_overrides or None

    async def __call__(self, inputs: AgentInputs) -> AgentOutputs:
        """Send the case as a new conversation and return the full transcript.

        Raises ValueError when the last message is not text; an AgentError from
        the agent propagates, so aevaluate records the row as errored.
        """
        *history, message = inputs["messages"]
        text = message["content"]
        if not isinstance(text, str):
            raise ValueError(settings.texts.last_not_user)
        turn = await self._agent.send(
            text,
            history=history,
            prompt_overrides=self._prompt_overrides,
            context=inputs["context"],
        )
        return AgentOutputs(
            text=turn.text,
            messages=[*inputs["messages"], *turn.messages],
            trajectory_reported=turn.trajectory_reported,
        )


__all__ = ["GoldenTarget"]
