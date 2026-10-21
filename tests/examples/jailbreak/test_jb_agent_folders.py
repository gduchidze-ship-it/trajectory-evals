"""The detector and assistant agent folders, and their sample datasets, load."""

from pathlib import Path

import pytest

from examples.jailbreak.responder import factory
from tests.examples.jailbreak.jb_fakes import EXAMPLE_DIR, FakeDetector, example_agent
from tests.fakes import FakeAgent, scripted
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.simulation import (
    RedteamSection,
    ScenarioSource,
    SimulationTarget,
    agent_table,
    load_personas,
    redteam_cases,
)

PERSONAS = {
    "benign_follow_up": "benign",
    "benign_frustrated": "benign",
    "crescendo_trust_builder": "attack",
    "roleplay_framer": "attack",
    "trigger_word_planter": "attack",
    "instruction_override": "attack",
    "data_exfiltration": "attack",
}


@pytest.mark.parametrize("name", ["detector", "assistant"])
def test_each_folder_builds_its_responder(name: str) -> None:
    """[bridge].options validate and every file they name is read."""
    config = example_agent(name)
    assert config.bridge is not None

    factory(
        config.bridge.options,
        config,
        detector=FakeDetector(),
        model_factory=lambda _: scripted(),
    )


def test_the_assistant_has_the_seven_jailbreak_personas() -> None:
    """Its own personas, read from their files, replace the library's namesakes."""
    config = example_agent("assistant")
    personas = load_personas(config)

    assert {name: personas[name].kind for name in PERSONAS} == PERSONAS
    for name in PERSONAS:
        own = (EXAMPLE_DIR / "assistant" / "personas" / f"{name}.md").read_text()
        assert personas[name].prompt == own


def test_the_assistant_red_teams_its_attack_personas() -> None:
    """[redteam] names only the jailbreak attackers, each once per attempt."""
    config = example_agent("assistant")
    redteam = agent_table(config, "redteam", RedteamSection)

    cases = redteam_cases(load_personas(config), names=redteam.personas, attempts=1)

    attackers = [name for name, kind in PERSONAS.items() if kind == "attack"]
    assert [case["inputs"]["persona"] for case in cases.examples()] == attackers


def test_the_assistant_simulation_tables_are_valid() -> None:
    """[simulation] and [redteam] validate, their files read and stop_when resolves."""
    config = example_agent("assistant")

    SimulationTarget(FakeAgent(), config=config, model_factory=lambda _: scripted())


def _sample(folder: str, name: str) -> Path:
    """Return the path of a committed sample dataset."""
    return EXAMPLE_DIR / folder / "datasets" / name


def test_the_detection_sample_is_labelled_golden_data() -> None:
    """Every sample row loads and carries an action on every channel routed apart."""
    examples = list(
        JsonlGoldenSource(_sample("detector", "detection.sample.jsonl")).examples()
    )

    actions = {example["outputs"]["action"] for example in examples}
    channels = {example["inputs"]["context"]["channel"] for example in examples}
    assert actions == {"ALLOWED", "BLOCKED", "REVIEW"}
    assert {"sms", "email", "submission_form"} <= channels


def test_the_scenario_sample_uses_the_assistants_personas() -> None:
    """Every sample scenario loads and names one of the assistant's personas."""
    source = ScenarioSource(_sample("assistant", "scenarios.sample.jsonl"))

    personas = {example["inputs"]["persona"] for example in source.examples()}

    assert personas <= PERSONAS.keys()
