"""Tests for optimizer settings and the [optimize] and [variants] tables."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.optimize.optimize_fakes import agent_folder, optimize_toml
from trajectory_evals.core.agent_config import AgentConfigError
from trajectory_evals.optimize.settings import (
    OptimizeSettings,
    optimize_table,
    settings,
    variant_tables,
)


def test_shipped_toml_loads() -> None:
    """The shipped config.toml loads into the module's settings."""
    assert OptimizeSettings() == settings


def test_environment_overrides_one_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TRAJECTORY_EVALS_OPTIMIZE_<SECTION>__<KEY> variable replaces one value."""
    monkeypatch.setenv("TRAJECTORY_EVALS_OPTIMIZE_DEFAULTS__PATIENCE", "5")

    loaded = OptimizeSettings()

    assert loaded.defaults.patience == 5
    assert loaded.defaults.iterations == settings.defaults.iterations


def test_a_template_missing_its_placeholders_fails_at_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reflection prompt that drops a placeholder the code fills in is refused."""
    monkeypatch.setenv(
        "TRAJECTORY_EVALS_OPTIMIZE_PROMPTS__REFLECTION_PROMPT", "{prompts} {failures}"
    )

    with pytest.raises(ValidationError, match="must use exactly the placeholders"):
        OptimizeSettings()


def test_absent_keys_take_the_config_defaults(tmp_path: Path) -> None:
    """Only prompts and the two required floors must be set."""
    table = (
        '[optimize]\nprompts = { system = "prompts/system.md" }\n'
        "floors = { pass_rate = 0.8, max_error_rate = 0.05 }\n"
    )
    defaults = settings.defaults

    section = optimize_table(agent_folder(tmp_path, table))

    assert section.reflection_model == defaults.reflection_model
    assert section.floors.min_judge_coverage == defaults.min_judge_coverage
    assert section.max_regression.attack_succeeded == 0.0
    assert (section.budget.iterations, section.budget.max_agent_calls) == (8, None)
    assert (section.split.train, section.split.val, section.split.holdout) == (
        0.5,
        0.25,
        0.25,
    )
    assert (section.redteam, section.seed, section.cost) == (True, 0, 1.0)


BAD_TABLES = {
    "no_table": ("", "prompts"),
    "typo": (optimize_toml(extra="iteration = 3\n"), "iteration: Extra"),
    "split_sum": (
        optimize_toml(extra="split = { train = 0.5, val = 0.5, holdout = 0.5 }\n"),
        "split ratios must add up to 1",
    ),
    "path_name": (
        '[optimize]\nprompts = { "../x" = "x.md" }\n'
        "floors = { pass_rate = 0.5, max_error_rate = 0.0 }\n",
        "prompts.../x",
    ),
}


@pytest.mark.parametrize(
    ("table", "reason"), BAD_TABLES.values(), ids=BAD_TABLES.keys()
)
def test_a_bad_optimize_table_names_the_table(
    tmp_path: Path, table: str, reason: str
) -> None:
    """A missing table, unknown key, bad split or path-like name is refused."""
    with pytest.raises(AgentConfigError, match=r"\[optimize\].*") as raised:
        optimize_table(agent_folder(tmp_path, table))

    assert reason in str(raised.value)


def test_variants_are_validated_against_the_baseline_prompts(tmp_path: Path) -> None:
    """A variant may only override the prompts [optimize].prompts names."""
    variants = '[variants.cheap]\nprompt_overrides = { system = "s.md" }\ncost = 0.4\n'
    config = agent_folder(tmp_path, optimize_toml() + variants)

    [(name, variant)] = variant_tables(config, optimize_table(config)).items()

    assert (name, variant.prompt_overrides, variant.cost) == (
        "cheap",
        {"system": "s.md"},
        0.4,
    )


BAD_VARIANTS = {
    "unknown_prompt": (
        '[variants.cheap]\nprompt_overrides = { greeting = "g.md" }\ncost = 0.4\n',
        "overrides prompts ['greeting']",
    ),
    "reserved_name": (
        '[variants.baseline]\nprompt_overrides = { system = "s.md" }\ncost = 0.4\n',
        "variant name 'baseline' is reserved",
    ),
    "no_cost": (
        '[variants.cheap]\nprompt_overrides = { system = "s.md" }\n',
        "cost",
    ),
}


@pytest.mark.parametrize(
    ("table", "reason"), BAD_VARIANTS.values(), ids=BAD_VARIANTS.keys()
)
def test_a_bad_variant_is_refused(tmp_path: Path, table: str, reason: str) -> None:
    """An unknown prompt, a reserved name or a missing cost raises naming [variants]."""
    config = agent_folder(tmp_path, optimize_toml() + table)

    with pytest.raises(AgentConfigError, match=r"\[variants\]") as raised:
        variant_tables(config, optimize_table(config))

    assert reason in str(raised.value)
