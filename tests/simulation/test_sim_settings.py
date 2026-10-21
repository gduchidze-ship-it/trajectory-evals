"""Tests for simulation settings and the agent.toml tables they validate."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.simulation.simulation_test_fakes import agent_folder
from trajectory_evals.core.agent_config import AgentConfigError
from trajectory_evals.simulation import RedteamSection, SimulationSection, agent_table
from trajectory_evals.simulation.settings import SimulationSettings, settings


def test_shipped_toml_loads() -> None:
    """config.toml and personas.toml load into the module's settings."""
    assert SimulationSettings() == settings


def test_environment_overrides_one_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TRAJECTORY_EVALS_SIMULATION_<SECTION>__<KEY> variable replaces one value."""
    monkeypatch.setenv("TRAJECTORY_EVALS_SIMULATION_DEFAULTS__STOP_TOKEN", "DONE")

    loaded = SimulationSettings()

    assert loaded.defaults.stop_token == "DONE"
    assert loaded.defaults.max_turns == settings.defaults.max_turns


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("TRAJECTORY_EVALS_SIMULATION_PROMPTS__STOP_INSTRUCTION", "Say bye."),
        ("TRAJECTORY_EVALS_SIMULATION_PERSONAS__CRESCENDO__PROMPT", "Escalate."),
        ("TRAJECTORY_EVALS_SIMULATION_PROMPTS__ATTACK_JUDGE", "{policy} {reply}"),
    ],
    ids=["stop_instruction", "persona_prompt", "attack_judge"],
)
def test_a_template_missing_its_placeholders_fails_at_load(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    """A prompt that drops a placeholder the code fills in is refused."""
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError, match="must use exactly the placeholders"):
        SimulationSettings()


def test_absent_tables_take_the_config_defaults(tmp_path: Path) -> None:
    """An agent without [simulation] or [redteam] gets the package defaults."""
    config = agent_folder(tmp_path)
    defaults = settings.defaults

    simulation = agent_table(config, "simulation", SimulationSection)
    redteam = agent_table(config, "redteam", RedteamSection)

    assert simulation == SimulationSection(
        user_model=defaults.user_model,
        max_turns=defaults.max_turns,
        stop_token=defaults.stop_token,
    )
    assert (simulation.user_prompt, simulation.stop_when) == (None, None)
    assert (redteam.attempts, redteam.judge_model) == (
        defaults.attempts,
        defaults.judge_model,
    )
    assert (redteam.brief, redteam.policy, redteam.personas) == (None, None, None)


BAD_TABLES = {
    "zero_turns": ("[simulation]\nmax_turns = 0", "simulation", "max_turns"),
    "typo": ("[simulation]\nmax_turn = 3", "simulation", "max_turn: Extra"),
    "bad_hook": ('[simulation]\nstop_when = "no colon"', "simulation", "stop_when"),
    "zero_attempts": ("[redteam]\nattempts = 0", "redteam", "attempts"),
    "no_personas": ("[redteam]\npersonas = []", "redteam", "personas"),
}


@pytest.mark.parametrize(
    ("toml", "table", "field"), BAD_TABLES.values(), ids=BAD_TABLES
)
def test_a_bad_table_names_the_file_table_and_field(
    tmp_path: Path, toml: str, table: str, field: str
) -> None:
    """A bad [simulation] or [redteam] value raises AgentConfigError saying where."""
    config = agent_folder(tmp_path, toml + "\n")
    schema = SimulationSection if table == "simulation" else RedteamSection

    with pytest.raises(AgentConfigError, match=f"agent.toml: \\[{table}\\] {field}"):
        agent_table(config, table, schema)
