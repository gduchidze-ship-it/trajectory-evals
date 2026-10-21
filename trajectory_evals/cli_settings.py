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
    """The program name."""

    prog: Text


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
    run_limit: PositiveInt
    harvest_file: Text
    stamp_format: Text


class SegmentSettings(_Section):
    """The metadata, output or input keys each command's pass_rate is segmented by."""

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
    options: Text


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
    url: Text
    dataset: Text
    project: Text
    filter: Text
    key: Text
    score: Text


class HelpSettings(_Section):
    """The help text of every command and option, and the metavars."""

    description: Text
    check: Text
    serve: Text
    eval: Text
    simulate: Text
    redteam: Text
    optimize: Text
    sweep: Text
    langsmith: Text
    pull: Text
    push: Text
    harvest: Text
    score: Text
    agent: Text
    ping: Text
    host: Text
    port: Text
    url: Text
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
    upload_as_experiments: Text
    include: Text
    dataset: Text
    pull_out: Text
    push_dataset: Text
    harvest_out: Text
    project: Text
    since_days: Text
    run_limit: Text
    filter: Text
    only_errors: Text
    feedback_key: Text
    feedback_below: Text
    dry_run: Text
    metavars: HelpMetavars


class TextSettings(_Section):
    """The CLI's own output lines and its error messages."""

    error_line: Text
    no_langsmith_key: Text
    data_not_found: Text
    folder_not_found: Text
    no_rows: Text
    not_positive_int: Text
    bad_model: Text
    model_failed: Text
    agent_failed: Text
    no_policy: Text
    unknown_personas: Text
    no_project: Text
    open_bind: Text
    needs_url: Text
    low_coverage: Text
    asr_line: Text
    results_line: Text
    agent_line: Text
    tables_line: Text
    dataset_line: Text
    card_line: Text
    ping_line: Text
    optimize_line: Text
    sweep_line: Text
    variant_line: Text
    no_reason: Text
    wrote_line: Text
    pushed_line: Text
    runs_line: Text
    key_line: Text


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
        rates = {"defended", "asr", "coverage", "error_rate"}
        _require_placeholders(
            {
                "defaults.serve_url": (defaults.serve_url, {"host", "port"}),
                "defaults.experiment_prefix": (
                    defaults.experiment_prefix,
                    {"agent", "command"},
                ),
                "defaults.dataset_name": (defaults.dataset_name, {"prefix", "source"}),
                "defaults.harvest_file": (defaults.harvest_file, {"agent", "stamp"}),
                "texts.error_line": (texts.error_line, {"prog", "error"}),
                "texts.no_langsmith_key": (texts.no_langsmith_key, {"env"}),
                "texts.data_not_found": (texts.data_not_found, {"path"}),
                "texts.folder_not_found": (texts.folder_not_found, {"path"}),
                "texts.no_rows": (texts.no_rows, {"path"}),
                "texts.not_positive_int": (texts.not_positive_int, {"text"}),
                "texts.bad_model": (texts.bad_model, {"name"}),
                "texts.model_failed": (texts.model_failed, {"name", "error"}),
                "texts.agent_failed": (texts.agent_failed, {"name", "error"}),
                "texts.no_policy": (texts.no_policy, set()),
                "texts.unknown_personas": (
                    texts.unknown_personas,
                    {"path", "names", "known"},
                ),
                "texts.no_project": (texts.no_project, set()),
                "texts.open_bind": (texts.open_bind, {"host"}),
                "texts.needs_url": (texts.needs_url, {"url"}),
                "texts.low_coverage": (texts.low_coverage, {"key", "coverage"}),
                "texts.asr_line": (texts.asr_line, rates),
                "texts.results_line": (texts.results_line, {"path"}),
                "texts.agent_line": (texts.agent_line, {"name", "root"}),
                "texts.tables_line": (texts.tables_line, set()),
                "texts.dataset_line": (texts.dataset_line, {"path", "count"}),
                "texts.card_line": (texts.card_line, set()),
                "texts.ping_line": (texts.ping_line, {"text"}),
                "texts.optimize_line": (
                    texts.optimize_line,
                    {"verdict", "reason", "best", "stop"},
                ),
                "texts.sweep_line": (texts.sweep_line, {"verdict", "reason", "winner"}),
                "texts.variant_line": (
                    texts.variant_line,
                    {"id", "cost", "pass_rate", "verdict", "reason"},
                ),
                "texts.no_reason": (texts.no_reason, set()),
                "texts.wrote_line": (texts.wrote_line, {"path"}),
                "texts.pushed_line": (texts.pushed_line, {"path", "name"}),
                "texts.runs_line": (texts.runs_line, {"runs"}),
                "texts.key_line": (
                    texts.key_line,
                    {"key", "scored", "skipped", "errors", "written"},
                ),
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
