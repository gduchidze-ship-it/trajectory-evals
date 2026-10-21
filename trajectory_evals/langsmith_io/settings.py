"""LangSmith client settings, and the agent.toml [langsmith] table they govern."""

from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, PositiveInt, StringConstraints, model_validator
from pydantic_settings import SettingsConfigDict

from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.settings import (
    PackageSettings,
    Text,
    _require_placeholders,
    _Section,
)
from trajectory_evals.simulation.settings import agent_table

PACKAGE_DIR = Path(__file__).parent
CONFIG_PATH = PACKAGE_DIR / "config.toml"

DottedPath = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[^.\s]+(\.[^.\s]+)*$")
]


class DefaultSettings(_Section):
    """[langsmith] defaults, and how many runs one query page asks for."""

    namespace: Text
    messages_key: DottedPath
    since_days: PositiveInt
    page_size: PositiveInt = Field(le=1000)


class KeySettings(_Section):
    """The policy judge's feedback key and the feedback-id namespace."""

    policy_violation: Text
    feedback_namespace: Text


class DataSettings(_Section):
    """Example metadata keys LangSmith adds itself, which a pull drops."""

    server_keys: tuple[Text, ...]


class TextSettings(_Section):
    """Messages for a refused write or call, and the progress log formats."""

    exists: Text
    not_ignored: Text
    below_needs_key: Text
    below_not_finite: Text
    under_review: Text
    sampled_client: Text
    harvested: Text
    scored: Text
    evaluator_failed: Text
    feedback_failed: Text
    dry_run: Text


class LangSmithIOSettings(PackageSettings):
    """All LangSmith client settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH, env_prefix="TRAJECTORY_EVALS_LANGSMITH_IO_"
    )

    defaults: DefaultSettings
    keys: KeySettings
    data: DataSettings
    texts: TextSettings

    @model_validator(mode="after")
    def _check_templates(self) -> Self:
        """Reject a template whose placeholders differ from those the code fills in."""
        texts = self.texts
        _require_placeholders(
            {
                "texts.exists": (texts.exists, {"path"}),
                "texts.not_ignored": (texts.not_ignored, {"path", "data_dir"}),
                "texts.below_needs_key": (texts.below_needs_key, set()),
                "texts.below_not_finite": (texts.below_not_finite, {"value"}),
                "texts.under_review": (texts.under_review, {"case_id", "flag"}),
                "texts.sampled_client": (texts.sampled_client, {"rate"}),
                "texts.harvested": (texts.harvested, set()),
                "texts.scored": (texts.scored, set()),
                "texts.evaluator_failed": (texts.evaluator_failed, set()),
                "texts.feedback_failed": (texts.feedback_failed, set()),
                "texts.dry_run": (texts.dry_run, set()),
            }
        )
        return self


class LangSmithSection(_Section):
    """agent.toml [langsmith]: where the agent traces, and how its runs read.

    project is the tracing project its production or dev runs land in, and
    dataset_prefix starts the names of the datasets its files are pushed to.
    namespace scopes pushed example ids. inputs_key and outputs_key are dotted
    paths to the messages inside a root run's inputs and outputs. since_days is
    how far back a harvest or score reads by default.
    """

    project: Text | None = None
    dataset_prefix: Text | None = None
    namespace: Text = Field(default_factory=lambda: settings.defaults.namespace)
    inputs_key: DottedPath = Field(
        default_factory=lambda: settings.defaults.messages_key
    )
    outputs_key: DottedPath = Field(
        default_factory=lambda: settings.defaults.messages_key
    )
    since_days: PositiveInt = Field(
        default_factory=lambda: settings.defaults.since_days
    )


def langsmith_section(config: AgentConfig) -> LangSmithSection:
    """Return the agent's validated [langsmith] table; AgentConfigError if bad."""
    return agent_table(config, "langsmith", LangSmithSection)


# Annotated because the section defaults above read it before mypy reaches here.
settings: LangSmithIOSettings = LangSmithIOSettings()

__all__ = [
    "CONFIG_PATH",
    "PACKAGE_DIR",
    "DataSettings",
    "DefaultSettings",
    "DottedPath",
    "KeySettings",
    "LangSmithIOSettings",
    "LangSmithSection",
    "TextSettings",
    "langsmith_section",
    "settings",
]
