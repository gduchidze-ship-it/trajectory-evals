"""Optimizer settings, and the agent.toml [optimize] and [variants] tables."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    model_validator,
)
from pydantic_settings import SettingsConfigDict

from trajectory_evals.core.agent_config import AGENT_FILE, AgentConfig, AgentConfigError
from trajectory_evals.core.settings import (
    PackageSettings,
    RawText,
    Text,
    _require_placeholders,
    _Section,
)
from trajectory_evals.simulation.settings import agent_table

PACKAGE_DIR = Path(__file__).parent
CONFIG_PATH = PACKAGE_DIR / "config.toml"

Share = Annotated[float, Field(ge=0, le=1)]
# A prompt or variant name becomes a file or folder name, so it cannot hold a path.
Name = Annotated[str, StringConstraints(pattern=r"^\w[\w-]*$")]


class DefaultSettings(_Section):
    """What [optimize] uses for each key agent.toml leaves out."""

    reflection_model: Text
    cost: NonNegativeFloat
    min_judge_coverage: Share
    max_attack_succeeded_regression: NonNegativeFloat
    redteam: bool
    seed: int
    iterations: PositiveInt
    patience: PositiveInt
    min_delta: NonNegativeFloat
    failures_per_reflection: PositiveInt
    repetitions: PositiveInt
    train: Share
    val: Share
    holdout: Share


class RunSettings(_Section):
    """How candidates' experiments run, and the names of what a run writes."""

    max_concurrency: PositiveInt
    min_val_cases: PositiveInt
    excerpt_chars: PositiveInt
    run_name: Text
    dataset_name: Text
    experiment_prefix: Text
    eval_run_id: Text
    candidate_id: Text
    prompt_file: Text


class KeySettings(_Section):
    """Candidate ids, the case metadata read, and the names of runs and modes."""

    baseline: Text
    include: Text
    split: Text
    risk: Text
    high_risk: Text
    redteam: Text
    optimize: Text
    sweep: Text


class VerdictSettings(_Section):
    """A candidate's status, and the final verdict on a run."""

    baseline: Text
    accepted: Text
    rejected: Text
    invalid: Text
    passed: Text
    adopt: Text
    do_not_adopt: Text


class StopSettings(_Section):
    """Why the optimizer stopped proposing candidates."""

    iterations: Text
    patience: Text
    no_failures: Text
    max_agent_calls: Text


class ReasonSettings(_Section):
    """Why a candidate was not accepted or adopted."""

    separator: RawText
    pass_rate_floor: Text
    error_rate_floor: Text
    latency_floor: Text
    judge_coverage: Text
    min_delta: Text
    high_risk: Text
    attack_coverage: Text
    attack_succeeded: Text
    names: Text
    reflection_failed: Text
    not_above_baseline: Text
    no_winner: Text
    is_baseline: Text


class PromptSettings(_Section):
    """The reflection model's system message, and the data it is sent."""

    reflection_system: Text
    reflection_prompt: Text
    prompt_block: Text
    prompt_separator: RawText


class TextSettings(_Section):
    """Messages for bad configuration or data, and progress log formats."""

    table_error: Text
    split_sum: Text
    unknown_prompts: Text
    reserved_variant: Text
    bad_split: Text
    empty_split: Text
    partial_split: Text
    small_val: Text
    reflection_failed: Text
    evaluated: Text
    verdict: Text
    stopped: Text
    final: Text


class OptimizeSettings(PackageSettings):
    """All optimizer settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH, env_prefix="TRAJECTORY_EVALS_OPTIMIZE_"
    )

    defaults: DefaultSettings
    run: RunSettings
    keys: KeySettings
    verdicts: VerdictSettings
    stops: StopSettings
    reasons: ReasonSettings
    prompts: PromptSettings
    texts: TextSettings

    @model_validator(mode="after")
    def _check_templates(self) -> Self:
        """Reject a template whose placeholders differ from those the code fills in."""
        run, reasons, texts = self.run, self.reasons, self.texts
        value_floor = {"value", "floor"}
        candidate = {"agent", "mode", "candidate", "split"}
        _require_placeholders(
            {
                "run.run_name": (run.run_name, {"agent", "mode"}),
                "run.dataset_name": (run.dataset_name, {"agent", "split"}),
                "run.experiment_prefix": (run.experiment_prefix, candidate),
                "run.eval_run_id": (run.eval_run_id, {"candidate", "split"}),
                "run.candidate_id": (run.candidate_id, {"number"}),
                "run.prompt_file": (run.prompt_file, {"name"}),
                "reasons.pass_rate_floor": (reasons.pass_rate_floor, value_floor),
                "reasons.error_rate_floor": (reasons.error_rate_floor, value_floor),
                "reasons.latency_floor": (reasons.latency_floor, value_floor),
                "reasons.judge_coverage": (
                    reasons.judge_coverage,
                    {"key", "value", "floor"},
                ),
                "reasons.min_delta": (
                    reasons.min_delta,
                    {"value", "best", "min_delta"},
                ),
                "reasons.high_risk": (reasons.high_risk, {"cases"}),
                "reasons.attack_coverage": (reasons.attack_coverage, {"value"}),
                "reasons.attack_succeeded": (
                    reasons.attack_succeeded,
                    {"value", "baseline", "limit"},
                ),
                "reasons.names": (reasons.names, {"names", "expected"}),
                "reasons.reflection_failed": (reasons.reflection_failed, {"error"}),
                "reasons.not_above_baseline": (
                    reasons.not_above_baseline,
                    {"value", "baseline"},
                ),
                "prompts.reflection_prompt": (
                    self.prompts.reflection_prompt,
                    {"prompts", "failures", "rejected"},
                ),
                "prompts.prompt_block": (self.prompts.prompt_block, {"name", "text"}),
                "texts.table_error": (texts.table_error, {"path", "table", "message"}),
                "texts.unknown_prompts": (
                    texts.unknown_prompts,
                    {"variant", "names"},
                ),
                "texts.reserved_variant": (texts.reserved_variant, {"name"}),
                "texts.bad_split": (texts.bad_split, {"allowed", "labels"}),
                "texts.empty_split": (texts.empty_split, {"splits"}),
            }
        )
        return self


class Floors(_Section):
    """[optimize].floors: what a candidate's val run must clear before cost counts.

    min_judge_coverage applies to each judge's share of scored rows among those
    it was due to score, so an outage cannot pass as a clean run.
    """

    pass_rate: Share
    max_error_rate: Share
    latency_p95_ms: PositiveFloat | None = None
    min_judge_coverage: Share = Field(
        default_factory=lambda: settings.defaults.min_judge_coverage
    )


class Regression(_Section):
    """[optimize].max_regression: how far red-team scores may worsen from baseline."""

    attack_succeeded: NonNegativeFloat = Field(
        default_factory=lambda: settings.defaults.max_attack_succeeded_regression
    )


class Budget(_Section):
    """[optimize].budget: when the optimizer stops, and how candidates are run.

    max_agent_calls None means no cap; iterations and patience still stop it.
    """

    iterations: PositiveInt = Field(
        default_factory=lambda: settings.defaults.iterations
    )
    patience: PositiveInt = Field(default_factory=lambda: settings.defaults.patience)
    min_delta: NonNegativeFloat = Field(
        default_factory=lambda: settings.defaults.min_delta
    )
    max_agent_calls: PositiveInt | None = None
    failures_per_reflection: PositiveInt = Field(
        default_factory=lambda: settings.defaults.failures_per_reflection
    )
    repetitions: PositiveInt = Field(
        default_factory=lambda: settings.defaults.repetitions
    )


class Split(_Section):
    """[optimize].split: the shares of cases a hashed-id split puts in each split."""

    train: Share = Field(default_factory=lambda: settings.defaults.train)
    val: Share = Field(default_factory=lambda: settings.defaults.val)
    holdout: Share = Field(default_factory=lambda: settings.defaults.holdout)

    @model_validator(mode="after")
    def _check_sum(self) -> Self:
        """Reject shares that do not add up to 1."""
        if not math.isclose(self.train + self.val + self.holdout, 1):
            raise ValueError(settings.texts.split_sum)
        return self


class OptimizeSection(_Section):
    """agent.toml [optimize]: the prompts optimized, the gates and the budget.

    prompts maps each prompt name the agent takes as an override to its file in
    the agent folder; those files are the baseline and are never written. cost is
    the baseline's declared cost, which the sweep compares with each variant's.
    """

    prompts: dict[Name, Text] = Field(min_length=1)
    reflection_model: Text = Field(
        default_factory=lambda: settings.defaults.reflection_model
    )
    cost: NonNegativeFloat = Field(default_factory=lambda: settings.defaults.cost)
    floors: Floors
    max_regression: Regression = Field(default_factory=Regression)
    redteam: bool = Field(default_factory=lambda: settings.defaults.redteam)
    budget: Budget = Field(default_factory=Budget)
    seed: int = Field(default_factory=lambda: settings.defaults.seed)
    split: Split = Field(default_factory=Split)


class VariantSection(_Section):
    """agent.toml [variants.<name>]: prompt files that replace the baseline's, and cost.

    Prompts a variant does not override stay the baseline's.
    """

    prompt_overrides: dict[Name, Text] = Field(min_length=1)
    cost: NonNegativeFloat


def optimize_table(config: AgentConfig) -> OptimizeSection:
    """Return the agent's [optimize] table; AgentConfigError if absent or invalid."""
    return agent_table(config, "optimize", OptimizeSection)


def variant_tables(
    config: AgentConfig, section: OptimizeSection
) -> dict[str, VariantSection]:
    """Return the agent's [variants.<name>] tables by name, none if absent.

    Raises AgentConfigError for an invalid table, a variant named like a
    candidate id, or one overriding a prompt [optimize].prompts does not name.
    """
    variants = agent_table(config, "variants", dict[Name, VariantSection])
    texts, keys = settings.texts, settings.keys
    for name, variant in variants.items():
        if name in {keys.baseline, keys.include}:
            raise _table_error(config, texts.reserved_variant.format(name=name))
        unknown = sorted(variant.prompt_overrides.keys() - section.prompts.keys())
        if unknown:
            message = texts.unknown_prompts.format(variant=name, names=unknown)
            raise _table_error(config, message)
    return variants


def read_prompts(config: AgentConfig, files: Mapping[str, str]) -> dict[str, str]:
    """Return each named prompt's text, read from its file in the agent folder."""
    return {name: config.read_text(path) for name, path in files.items()}


def _table_error(config: AgentConfig, message: str) -> AgentConfigError:
    """Return the error for the agent's [variants] table."""
    path = config.root / AGENT_FILE
    text = settings.texts.table_error
    return AgentConfigError(text.format(path=path, table="variants", message=message))


# Annotated because the section defaults above read it before mypy reaches here.
settings: OptimizeSettings = OptimizeSettings()

__all__ = [
    "CONFIG_PATH",
    "PACKAGE_DIR",
    "Budget",
    "DefaultSettings",
    "Floors",
    "KeySettings",
    "Name",
    "OptimizeSection",
    "OptimizeSettings",
    "PromptSettings",
    "ReasonSettings",
    "Regression",
    "RunSettings",
    "Split",
    "StopSettings",
    "TextSettings",
    "VariantSection",
    "VerdictSettings",
    "optimize_table",
    "read_prompts",
    "settings",
    "variant_tables",
]
