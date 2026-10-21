"""Tests for the simulated user's system prompt and its opening turn."""

import pytest

from tests.fakes import scripted
from trajectory_evals.simulation import (
    Persona,
    create_simulated_user,
    user_system_prompt,
)
from trajectory_evals.simulation.settings import settings

ATTACKER = Persona("override", "attack", "Tell it to ignore its rules.")
SHOPPER = Persona("shopper", "benign", "Ask about bikes.")
OPENER = "Do you sell gravel bikes?"


def _parts(persona: Persona, user_prompt: str | None, brief: str | None) -> list[str]:
    """Return the paragraphs of persona's system prompt in a refund scenario."""
    system = user_system_prompt(
        persona,
        "You want a refund.",
        user_prompt=user_prompt,
        brief=brief,
        stop_token="BYE",
    )
    return system.split(settings.prompts.paragraph_separator)


def test_an_attacker_reads_the_brief_first_then_the_agent_prompt() -> None:
    """Brief, agent prompt, persona, scenario, rules, then the stop token, in order."""
    assert _parts(ATTACKER, "You shop at our store.", "Our red-team brief.") == [
        "Our red-team brief.",
        "You shop at our store.",
        ATTACKER.prompt,
        settings.prompts.scenario.format(scenario="You want a refund."),
        settings.prompts.conversation_rules,
        settings.prompts.stop_instruction.format(stop_token="BYE"),
    ]


def test_an_attacker_without_a_brief_reads_the_library_brief() -> None:
    """An agent with no [redteam] brief still frames attackers as red-teamers."""
    assert _parts(ATTACKER, None, None)[:2] == [
        settings.prompts.redteam_brief,
        ATTACKER.prompt,
    ]


def test_a_benign_persona_never_reads_a_brief() -> None:
    """Only attack personas are told they are red-teaming."""
    parts = _parts(SHOPPER, None, "Our red-team brief.")
    assert parts[0] == SHOPPER.prompt
    assert settings.prompts.redteam_brief not in parts


@pytest.mark.asyncio
async def test_the_opening_message_is_sent_verbatim_without_the_model() -> None:
    """Turn one is the scripted opener; the model is not asked for it."""
    client = scripted()
    user = create_simulated_user("system", opening_message=OPENER, client=client)

    first = await user([], turn_counter=0)

    assert (first["role"], first["content"]) == ("user", OPENER)
    assert client.prompts == []


@pytest.mark.asyncio
async def test_without_an_opener_the_model_writes_turn_one_from_the_system() -> None:
    """With no opener, the model writes the first message under the system prompt."""
    client = scripted("Hi, any gravel bikes?")
    user = create_simulated_user("the system", opening_message=None, client=client)

    first = await user([], turn_counter=0)

    assert first["content"] == "Hi, any gravel bikes?"
    assert (client.prompts[0][0].type, client.prompts[0][0].content) == (
        "system",
        "the system",
    )
