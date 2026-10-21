"""Tests for the target that sends a golden case's conversation to an Agent."""

import pytest

from tests.fakes import AgentCall, FakeAgent
from tests.golden.golden_fakes import TOOL_CALL
from trajectory_evals.core.types import (
    AgentError,
    AgentInputs,
    AgentOutputs,
    AgentTurn,
    ChatCompletionMessage,
    ToolMessage,
)
from trajectory_evals.golden.target import GoldenTarget

HISTORY: list[ChatCompletionMessage] = [
    {"role": "user", "content": "Where is A1?"},
    {"role": "assistant", "content": "It shipped."},
]
LAST: ChatCompletionMessage = {"role": "user", "content": "Refund it."}
INPUTS = AgentInputs(messages=[*HISTORY, LAST], context={"channel": "sms"})


@pytest.mark.asyncio
async def test_sends_the_last_message_with_history_and_context() -> None:
    """Earlier messages are history, context goes along, and no overrides are sent."""
    agent = FakeAgent("Refunded.")

    await GoldenTarget(agent)(INPUTS)

    assert agent.calls == [
        AgentCall(
            text="Refund it.",
            context_id=None,
            history=tuple(HISTORY),
            prompt_overrides=None,
            context={"channel": "sms"},
        )
    ]


@pytest.mark.asyncio
async def test_outputs_are_the_inputs_then_the_turns_messages() -> None:
    """The transcript is every input message followed by the agent's turn."""
    turn = AgentTurn(
        text="Refunded.",
        messages=(
            {"role": "assistant", "content": "", "tool_calls": [TOOL_CALL]},
            ToolMessage(role="tool", content="ok", tool_call_id="call-1"),
            {"role": "assistant", "content": "Refunded."},
        ),
        context_id="ctx-1",
        trajectory_reported=False,
    )

    outputs = await GoldenTarget(FakeAgent(turn))(INPUTS)

    assert outputs == AgentOutputs(
        text="Refunded.",
        messages=[*HISTORY, LAST, *turn.messages],
        trajectory_reported=False,
    )


@pytest.mark.parametrize(
    ("overrides", "sent"),
    [({"system": "Be brief."}, {"system": "Be brief."}), ({}, None)],
    ids=["forwarded", "empty_is_none"],
)
@pytest.mark.asyncio
async def test_prompt_overrides_go_with_every_case(
    overrides: dict[str, str], sent: dict[str, str] | None
) -> None:
    """Overrides are sent on each case; an empty mapping is sent as None."""
    agent = FakeAgent("a", "b")
    target = GoldenTarget(agent, prompt_overrides=overrides)

    await target(INPUTS)
    await target(INPUTS)

    assert [call.prompt_overrides for call in agent.calls] == [sent, sent]


@pytest.mark.asyncio
async def test_agent_errors_propagate() -> None:
    """An AgentError reaches aevaluate, which records an errored row."""

    def fail(call: AgentCall) -> str:
        """Fail as an agent that ignored its overrides would."""
        raise AgentError("prompt_overrides not applied")

    with pytest.raises(AgentError):
        await GoldenTarget(FakeAgent(respond=fail))(INPUTS)
