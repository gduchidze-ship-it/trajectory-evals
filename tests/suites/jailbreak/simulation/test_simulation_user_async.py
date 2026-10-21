"""Tests for the simulated customer that opens with a real message."""

import pytest

from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.simulation import PERSONAS, create_simulated_user
from tests.suites.jailbreak.simulation.simulation_fakes import scripted

OPENER = "Is the 2021 Civic still there?"


@pytest.mark.asyncio
async def test_first_turn_is_the_opening_message_verbatim() -> None:
    """Turn one is the real opener, sent without calling the model."""
    client = scripted()
    user = create_simulated_user(
        persona=PERSONAS["benign_follow_up"],
        opening_message=OPENER,
        channel="sms",
        max_turns=6,
        client=client,
    )
    first = await user([], turn_counter=0)
    assert (first["role"], first["content"]) == ("user", OPENER)
    assert client.prompts == []


@pytest.mark.parametrize(
    ("channel", "register"),
    [
        ("sms", settings.simulation.sms_register),
        ("email", settings.simulation.email_register),
        ("submission_form", settings.simulation.email_register),
    ],
)
@pytest.mark.asyncio
async def test_later_turns_play_the_persona_in_the_channel_register(
    channel: str, register: str
) -> None:
    """Later turns come from the model told its persona, register and turn budget."""
    client = scripted("Can I drive it Saturday?")
    persona = PERSONAS["roleplay_framer"]
    user = create_simulated_user(
        persona=persona,
        opening_message=OPENER,
        channel=channel,
        max_turns=4,
        client=client,
    )
    trajectory = [
        {"role": "user", "content": OPENER},
        {"role": "assistant", "content": "It is."},
    ]
    reply = await user(trajectory, turn_counter=1)
    assert (reply["role"], reply["content"]) == ("user", "Can I drive it Saturday?")
    budget = (
        "You have 3 messages after the opener; make your explicit ask by your last one."
    )
    assert [(message.type, message.content) for message in client.prompts[0]] == [
        ("system", f"{persona.system_prompt}\n\n{register}\n\n{budget}"),
        ("ai", OPENER),
        ("human", "It is."),
    ]
