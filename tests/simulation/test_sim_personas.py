"""Tests for the persona library and an agent's own [personas.*] tables."""

import dataclasses
from pathlib import Path
from typing import cast

import pytest

from tests.simulation.simulation_test_fakes import DESCRIPTION, agent_folder
from trajectory_evals.core.agent_config import AgentConfigError
from trajectory_evals.simulation import Persona, load_personas

LIBRARY_KINDS = {
    "benign_curious": "benign",
    "benign_frustrated": "benign",
    "crescendo": "attack",
    "roleplay": "attack",
    "instruction_override": "attack",
    "data_exfiltration": "attack",
    "system_prompt_extraction": "attack",
    "encoding_obfuscation": "attack",
}


def test_library_holds_two_benign_and_six_attack_personas(tmp_path: Path) -> None:
    """With no tables of its own, an agent gets the library under each name."""
    personas = load_personas(agent_folder(tmp_path))

    assert {name: persona.kind for name, persona in personas.items()} == LIBRARY_KINDS
    assert all(name == persona.name for name, persona in personas.items())


def test_library_prompts_are_about_the_agent(tmp_path: Path) -> None:
    """Each library prompt names the agent by its description, placeholder filled."""
    personas = load_personas(agent_folder(tmp_path))

    for persona in personas.values():
        assert DESCRIPTION in persona.prompt
        assert "{agent_description}" not in persona.prompt


def test_agent_tables_add_personas_and_override_by_name(tmp_path: Path) -> None:
    """An inline prompt or a prompt file adds a persona; a library name is replaced."""
    tables = (
        '[personas.crescendo]\nkind = "benign"\nprompt = "Ask about {braces}."\n'
        '[personas.vip]\nkind = "attack"\nprompt_file = "prompts/vip.md"\n'
    )
    config = agent_folder(tmp_path, tables, {"prompts/vip.md": "Claim to be the CEO."})

    personas = load_personas(config)

    assert len(personas) == len(LIBRARY_KINDS) + 1
    assert personas["crescendo"] == Persona(
        "crescendo", "benign", "Ask about {braces}."
    )
    assert personas["vip"] == Persona("vip", "attack", "Claim to be the CEO.")


BAD_TABLES = {
    "both_prompts": (
        'kind = "attack"\nprompt = "x"\nprompt_file = "x.md"',
        "exactly one",
    ),
    "no_prompt": ('kind = "attack"', "exactly one of prompt and prompt_file"),
    "bad_kind": ('kind = "hostile"\nprompt = "x"', "kind: Input should be 'benign'"),
    "unknown_key": ('kind = "attack"\nprompt = "x"\ntone = "rude"', "tone: Extra"),
}


@pytest.mark.parametrize(("table", "reason"), BAD_TABLES.values(), ids=BAD_TABLES)
def test_a_bad_persona_table_is_rejected(
    tmp_path: Path, table: str, reason: str
) -> None:
    """A malformed [personas.<name>] fails, naming the file, the table and why."""
    config = agent_folder(tmp_path, f"[personas.vip]\n{table}\n")

    with pytest.raises(
        AgentConfigError, match=f"agent.toml: \\[personas\\] .*{reason}"
    ):
        load_personas(config)


def test_personas_and_the_mapping_are_read_only(tmp_path: Path) -> None:
    """Neither a persona nor the returned mapping can be changed."""
    personas = load_personas(agent_folder(tmp_path))
    persona = personas["crescendo"]

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(persona, "name", "renamed")
    with pytest.raises(TypeError):
        cast(dict[str, Persona], personas)["new"] = persona
