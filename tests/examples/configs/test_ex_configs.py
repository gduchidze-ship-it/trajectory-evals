"""Each example folder loads, and every table, file and dataset it holds is valid."""

from collections.abc import Callable
from pathlib import Path
from pkgutil import resolve_name
from typing import Any

import pytest
from pydantic import BaseModel

from examples.support_agent.responder import TOOLS, SupportOptions
from tests.fakes import scripted
from trajectory_evals.connect.sequence_runner import SequenceRunnerOptions
from trajectory_evals.core.agent_config import load_agent
from trajectory_evals.core.protocols import CaseSource
from trajectory_evals.core.settings import REPO_ROOT
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.golden.graders import build_graders
from trajectory_evals.simulation import (
    RedteamSection,
    ScenarioSource,
    SimulationSection,
    agent_table,
    load_personas,
    redteam_cases,
)

EXAMPLES = REPO_ROOT / "examples"
FOLDERS = ("support_agent", "sequence_runner_sales", "remote_a2a")
PROMPT_SUFFIX = ".md"
BRIDGE_OPTIONS: dict[str, type[BaseModel]] = {
    "support_agent": SupportOptions,
    "sequence_runner_sales": SequenceRunnerOptions,
}
SOURCES: dict[str, Callable[[Path], CaseSource]] = {
    "golden.sample.jsonl": JsonlGoldenSource,
    "scenarios.sample.jsonl": ScenarioSource,
}
SAMPLES = sorted(
    path for name in FOLDERS for path in (EXAMPLES / name).rglob("*.sample.jsonl")
)
SEGMENTS = {"risk", "difficulty", "split"}
SPLITS = {"train", "val", "holdout"}
SUPPORT_DATASETS = EXAMPLES / "support_agent" / "datasets"


def _prompt_files(value: Any) -> list[str]:
    """Return every .md path among a table's values, at any depth."""
    if isinstance(value, str):
        return [value] if value.endswith(PROMPT_SUFFIX) else []
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return []
    return [path for item in value for path in _prompt_files(item)]


@pytest.mark.parametrize("name", FOLDERS)
def test_tables_validate_with_the_packages_that_own_them(name: str) -> None:
    """[simulation], [redteam], [personas] and [judges] build without a model call."""
    config = load_agent(EXAMPLES / name)

    agent_table(config, "simulation", SimulationSection)
    redteam = agent_table(config, "redteam", RedteamSection)
    personas = load_personas(config)
    redteam_cases(personas, names=redteam.personas, attempts=redteam.attempts)
    build_graders(config, model_factory=lambda _: scripted(), criteria=False)

    assert redteam.policy is not None


@pytest.mark.parametrize("name", FOLDERS)
def test_every_prompt_file_named_exists_in_the_folder(name: str) -> None:
    """Each .md path in any table reads from inside the agent folder."""
    config = load_agent(EXAMPLES / name)
    paths = _prompt_files(dict(config.sections))

    assert paths
    for path in paths:
        assert config.read_text(path).strip(), path


@pytest.mark.parametrize("name", BRIDGE_OPTIONS)
def test_bridge_responders_resolve_and_their_options_validate(name: str) -> None:
    """The responder factory imports, and [bridge].options fits its schema."""
    config = load_agent(EXAMPLES / name)
    assert config.bridge is not None

    assert callable(resolve_name(config.bridge.responder))
    BRIDGE_OPTIONS[name].model_validate(config.bridge.options)


@pytest.mark.parametrize("path", SAMPLES, ids=[path.name for path in SAMPLES])
def test_every_sample_dataset_parses_with_its_source(path: Path) -> None:
    """Each committed sample is read by the source its name says."""
    examples = list(SOURCES[path.name](path).examples())

    assert examples


def test_golden_rows_are_segmented_and_split() -> None:
    """Every golden row has risk and difficulty, and the splits cover all three."""
    cases = list(JsonlGoldenSource(SUPPORT_DATASETS / "golden.sample.jsonl").cases())

    assert all(case.metadata.keys() >= SEGMENTS for case in cases)
    assert {case.metadata["split"] for case in cases} == SPLITS


def test_expected_tools_are_tools_the_support_agent_has() -> None:
    """Every tool a sample expects, by name or in a trajectory, is in TOOLS."""
    sources = (
        JsonlGoldenSource(SUPPORT_DATASETS / "golden.sample.jsonl"),
        ScenarioSource(SUPPORT_DATASETS / "scenarios.sample.jsonl"),
    )
    expected = [
        example["outputs"] for source in sources for example in source.examples()
    ]
    named = {tool for outputs in expected for tool in outputs.get("tools", [])}
    called = {
        call["function"]["name"]
        for outputs in expected
        for message in outputs.get("trajectory", [])
        for call in message.get("tool_calls", [])
    }

    assert named
    assert called
    assert named | called <= {tool.__name__ for tool in TOOLS}


def test_scenarios_use_personas_the_support_agent_has() -> None:
    """A scenario's persona is a library persona or one of the agent's own."""
    personas = load_personas(load_agent(EXAMPLES / "support_agent"))
    source = ScenarioSource(SUPPORT_DATASETS / "scenarios.sample.jsonl")

    assert {example["inputs"]["persona"] for example in source.examples()} <= set(
        personas
    )
