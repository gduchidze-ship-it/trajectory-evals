"""Tests for the persona registry."""

import dataclasses
from typing import cast

import pytest

from trajectory_evals.suites.jailbreak.simulation import PERSONAS, Persona

EXPECTED_KINDS = {
    "benign_follow_up": "benign",
    "benign_frustrated": "benign",
    "crescendo_trust_builder": "attack",
    "roleplay_framer": "attack",
    "trigger_word_planter": "attack",
    "instruction_override": "attack",
    "data_exfiltration": "attack",
}
CONVERSATION_RULES = (
    "first message",
    "exactly one customer message per turn",
    "never break character",
)
ATTACK_FRAMING = ("authorized red-team exercise", "manipulative pretext")


def test_registry_holds_every_persona_with_its_kind() -> None:
    """The registry names two benign personas and five attackers."""
    assert {name: persona.kind for name, persona in PERSONAS.items()} == EXPECTED_KINDS


def test_registry_keys_are_the_persona_names() -> None:
    """Each persona is registered under its own name."""
    assert all(name == persona.name for name, persona in PERSONAS.items())


@pytest.mark.parametrize("persona", list(PERSONAS.values()), ids=list(PERSONAS))
def test_every_prompt_states_the_conversation_rules(persona: Persona) -> None:
    """Each prompt continues the real opener, one message per turn, in character."""
    assert all(rule in persona.system_prompt for rule in CONVERSATION_RULES)


@pytest.mark.parametrize("persona", list(PERSONAS.values()), ids=list(PERSONAS))
def test_only_attackers_are_red_teamers_who_need_a_pretext(persona: Persona) -> None:
    """Attack prompts are authorized red-teaming that demands a pretext; benign not."""
    framed = [phrase in persona.system_prompt for phrase in ATTACK_FRAMING]
    assert framed == [persona.kind == "attack"] * len(ATTACK_FRAMING)


def test_frustrated_customer_asks_about_bots_and_opting_out() -> None:
    """The hard benign persona questions the bot and may opt out, without attacking."""
    prompt = PERSONAS["benign_frustrated"].system_prompt
    assert "bot" in prompt
    assert "stop receiving messages" in prompt


def test_personas_and_registry_are_read_only() -> None:
    """Neither a persona nor the registry can be changed at runtime."""
    persona = PERSONAS["benign_follow_up"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(persona, "name", "renamed")
    with pytest.raises(TypeError):
        cast(dict[str, Persona], PERSONAS)["new"] = persona
