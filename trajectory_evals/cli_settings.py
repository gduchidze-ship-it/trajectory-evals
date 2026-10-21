"""Command line settings: cli.toml values, overridable by environment."""

from pathlib import Path
from typing import Self

from pydantic import PositiveInt, model_validator
from pydantic_settings import SettingsConfigDict

from trajectory_evals.core.settings import (
    PackageSettings,
    Text,
    _require_placeholders,
    _require_unique,
    _Section,
)

PACKAGE_DIR = Path(__file__).parent
CONFIG_PATH = PACKAGE_DIR / "cli.toml"


class ProgramSettings(_Section):
    """The program name, the exit code for bad input, and where uploads go."""

    prog: Text
    input_error_exit_code: PositiveInt
    langsmith_api_key_env: Text
    dataset_namespace: Text


class DefaultSettings(_Section):
    """What a command uses when its option is not given."""

    max_concurrency: PositiveInt
    results_dir: Text
    host: Text
    port: PositiveInt
    serve_url: Text
    experiment_prefix: Text
    dataset_name: Text
    ping_message: Text
    persona_separator: Text


class SegmentSettings(_Section):
    """The metadata or output keys each command's summary segments pass_rate by."""

    eval: tuple[Text, ...]
    simulate: tuple[Text, ...]
    redteam: tuple[Text, ...]


class CheckSettings(_Section):
    """Where check finds an agent's datasets, and which hold scenarios."""

    datasets_dir: Text
    datasets_glob: Text
    scenario_marker: Text


class MetadataKeys(_Section):
    """The experiment metadata and meta.json key each run fact is recorded under."""

    agent: Text
    command: Text
    data: Text
    history: Text
    experiment: Text
    dataset: Text
    url: Text


class HelpMetavars(_Section):
    """What argparse shows in place of each kind of option value."""

    command: Text
    agent: Text
    count: Text
    path: Text
    directory: Text
    prefix: Text
    model: Text
    names: Text
    host: Text
    port: Text


class HelpSettings(_Section):
    """The help text of every command and option, and the metavars."""

    description: Text
    check: Text
    serve: Text
    eval: Text
    simulate: Text
    redteam: Text
    agent: Text
    ping: Text
    host: Text
    port: Text
    data: Text
    scenarios: Text
    local: Text
    limit: Text
    repetitions: Text
    concurrency: Text
    prefix: Text
    results_dir: Text
    criteria_model: Text
    max_turns: Text
    personas: Text
    attempts: Text
    metavars: HelpMetavars


class TextSettings(_Section):
    """The CLI's own output lines and its error messages."""

    error_line: Text
    no_langsmith_key: Text
    data_not_found: Text
    no_rows: Text
    not_positive_int: Text
    bad_model: Text
    open_bind: Text
    low_coverage: Text
    results_line: Text
    agent_line: Text
    tables_line: Text
    dataset_line: Text
    card_line: Text
    ping_line: Text


class CliSettings(PackageSettings):
    """All command line settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH, env_prefix="TRAJECTORY_EVALS_CLI_"
    )

    cli: ProgramSettings
    defaults: DefaultSettings
    segments: SegmentSettings
    check: CheckSettings
    metadata_keys: MetadataKeys
    help: HelpSettings
    texts: TextSettings

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        """Reject metadata keys that would overwrite each other."""
        _require_unique({"metadata keys": self.metadata_keys.model_dump().values()})
        return self

    @model_validator(mode="after")
    def _check_templates(self) -> Self:
        """Reject a template whose placeholders differ from those the code fills in."""
        defaults, texts = self.defaults, self.texts
        _require_placeholders(
            {
                "defaults.serve_url": (defaults.serve_url, {"host", "port"}),
                "defaults.experiment_prefix": (
                    defaults.experiment_prefix,
                    {"agent", "command"},
                ),
                "defaults.dataset_name": (defaults.dataset_name, {"prefix", "source"}),
                "texts.error_line": (texts.error_line, {"prog", "error"}),
                "texts.no_langsmith_key": (texts.no_langsmith_key, {"env"}),
                "texts.data_not_found": (texts.data_not_found, {"path"}),
                "texts.no_rows": (texts.no_rows, {"path"}),
                "texts.not_positive_int": (texts.not_positive_int, {"text"}),
                "texts.bad_model": (texts.bad_model, {"name"}),
                "texts.open_bind": (texts.open_bind, {"host"}),
                "texts.low_coverage": (texts.low_coverage, {"key", "coverage"}),
                "texts.results_line": (texts.results_line, {"path"}),
                "texts.agent_line": (texts.agent_line, {"name", "root"}),
                "texts.tables_line": (texts.tables_line, set()),
                "texts.dataset_line": (texts.dataset_line, {"path", "count"}),
                "texts.card_line": (texts.card_line, set()),
                "texts.ping_line": (texts.ping_line, {"text"}),
            }
        )
        return self


settings = CliSettings()

__all__ = [
    "CONFIG_PATH",
    "PACKAGE_DIR",
    "CheckSettings",
    "CliSettings",
    "DefaultSettings",
    "HelpMetavars",
    "HelpSettings",
    "MetadataKeys",
    "ProgramSettings",
    "SegmentSettings",
    "TextSettings",
    "settings",
]
