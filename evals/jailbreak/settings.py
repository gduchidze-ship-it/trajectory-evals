"""Jailbreak eval settings: config.toml values, overridable by environment."""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from string import Formatter
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from evals.jailbreak.types import ChatRole, PersonaKind

PACKAGE_DIR = Path(__file__).parent
CONFIG_PATH = PACKAGE_DIR / "config.toml"
REPO_ROOT = PACKAGE_DIR.resolve().parents[1]

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RawText = Annotated[str, StringConstraints(min_length=1)]


def _placeholders(template: str) -> set[str]:
    """Return the replacement field names of a str.format template."""
    return {name for _, name, _, _ in Formatter().parse(template) if name is not None}


def _repeated(names: Iterable[str]) -> list[str]:
    """Return each name that occurs more than once, sorted."""
    return sorted(name for name, count in Counter(names).items() if count > 1)


class _Section(BaseModel):
    """A frozen config.toml table that rejects unknown keys, so a typo fails."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PathSettings(_Section):
    """Where the labelled JSONL files live, relative to the repo root."""

    data_dir: Text
    data_glob: Text
    simulation_data: Text

    @property
    def data_path(self) -> Path:
        """Return the data directory resolved against the repo root."""
        return REPO_ROOT / self.data_dir


class CliMetadataKeys(_Section):
    """The experiment metadata key each simulation option is recorded under."""

    personas: Text
    sample: Text
    seed: Text
    max_turns: Text
    user_model: Text
    responder_model: Text
    judge_model: Text


class CliSettings(_Section):
    """The command's name, exit code on bad input, variables and metadata keys."""

    prog: Text
    input_error_exit_code: PositiveInt
    aws_region_env: Text
    default_aws_region: Text
    langsmith_api_key_env: Text
    openai_api_key_env: Text
    metadata_keys: CliMetadataKeys


class HelpMetavars(_Section):
    """What argparse shows in place of each kind of option value."""

    command: Text
    count: Text
    path: Text
    prefix: Text
    persona: Text
    model: Text


class HelpSettings(_Section):
    """The help text of every command and option, and the metavars."""

    description: Text
    detection: Text
    simulation: Text
    detection_data: Text
    limit: Text
    max_concurrency: Text
    repetitions: Text
    local: Text
    experiment_prefix: Text
    simulation_data: Text
    sample: Text
    seed: Text
    personas: Text
    persona_separator: RawText
    max_turns: Text
    user_model: Text
    responder_model: Text
    judge_model: Text
    metavars: HelpMetavars


class ReportLabels(_Section):
    """The label of each fixed line of the printed experiment report."""

    experiment: Text
    dataset: Text
    url: Text
    runs: Text


class LangSmithSettings(_Section):
    """Experiment defaults, the trace tag, errored rows and the printed report."""

    repetitions: PositiveInt
    ls_framework: Text
    errored_output_key: Literal["output"]
    target_errored_comment: Text
    ids_per_request: PositiveInt
    prompt_hash_separator: RawText
    local_suffix: RawText
    not_uploaded: Text
    undefined_score: Text
    score_decimals: NonNegativeInt
    report_line: Text
    report_labels: ReportLabels


class DataSettings(_Section):
    """JSONL row keys, example metadata keys, the example-id namespace and batches."""

    id_key: Text
    label_key: Text
    text_keys: tuple[Text, ...] = Field(min_length=1)
    turns_key: Text
    history_key: Text
    source_file_key: Text
    case_id_key: Text
    content_hash_key: Text
    example_id_namespace: Text
    update_batch_size: PositiveInt
    delete_batch_size: PositiveInt

    @property
    def message_keys(self) -> tuple[str, ...]:
        """Return the row keys that can hold the message, the turns key last."""
        return (*self.text_keys, self.turns_key)

    @property
    def row_keys(self) -> tuple[str, ...]:
        """Return every row key a case reads; the other keys become metadata."""
        return (self.id_key, self.label_key, self.history_key, *self.message_keys)

    @property
    def reserved_keys(self) -> tuple[str, ...]:
        """Return the metadata keys the data layer writes, which a row cannot use."""
        return (self.source_file_key, self.case_id_key, self.content_hash_key)


class ChannelSettings(_Section):
    """The row key holding a case's channel, and the channels code treats apart."""

    key: Text
    sms: Text
    conversation_only: Text
    opener_text_only: Text
    unknown: Text


class DetectionMetricSettings(_Section):
    """The key each ConfusionMatrix metric is reported under, in report order."""

    false_positive_rate: Text
    detection_rate: Text
    precision: Text
    f1: Text
    escalation_rate: Text
    fail_open_rate: Text
    review_block_rate: Text
    error_rate: Text


class DetectionSettings(_Section):
    """Detection defaults, feedback keys, the review comment and reported metrics."""

    max_concurrency: PositiveInt
    experiment_prefix: Text
    dataset_prefix: Text
    correct_key: Text
    escalated_key: Text
    failed_open_key: Text
    review_comment: Text
    metrics: DetectionMetricSettings

    @property
    def feedback_keys(self) -> tuple[str, ...]:
        """Return every row and summary key a detection experiment reports."""
        row_keys = (self.correct_key, self.escalated_key, self.failed_open_key)
        return (*row_keys, *self.metrics.model_dump().values())


class SimulationMetricSettings(_Section):
    """The key each simulation summary measure is reported under."""

    attack_blocked: Text
    benign_blocked: Text
    attack_turns_to_block: Text
    faithful: Text
    errored: Text


class SimulationSettings(_Section):
    """Simulation defaults, roles, keys, comments and prompt texts."""

    max_concurrency: PositiveInt
    experiment_prefix: Text
    dataset_prefix: Text
    sample: PositiveInt
    seed: int
    max_turns: PositiveInt
    opener_turn: Literal[1]
    agent_initiator: Text
    customer_role: ChatRole
    agent_role: ChatRole
    system_role: ChatRole
    source_case_id_key: Text
    persona_kind_key: Text
    correct_key: Text
    turns_to_block_key: Text
    attack_attempted_key: Text
    persona_faithful_key: Text
    opener_blocked_comment: Text
    benign_comment: Text
    not_blocked_comment: Text
    not_judged_comment: Text
    not_attempted_comment: Text
    paragraph_separator: RawText
    conversation_rules: Text
    red_team_brief: Text
    sms_register: Text
    email_register: Text
    turn_budget: Text
    deflection: Text
    sales_assistant_prompt: Text
    metrics: SimulationMetricSettings

    @property
    def feedback_keys(self) -> tuple[str, ...]:
        """Return every row and summary key a simulation experiment reports."""
        row_keys = (
            self.correct_key,
            self.turns_to_block_key,
            self.attack_attempted_key,
            self.persona_faithful_key,
        )
        return (*row_keys, *self.metrics.model_dump().values())


class ModelSettings(_Section):
    """OpenAI model names of the simulated customer, the responder and the judge."""

    user: Text
    responder: Text
    judge: Text


class TextSettings(_Section):
    """Messages for a missing prerequisite, a bad data row or a misused API."""

    no_aws_credentials: Text
    no_langsmith_key: Text
    no_openai_key: Text
    no_data: Text
    data_not_found: Text
    not_positive_int: Text
    no_message: Text
    not_an_object: Text
    reserved_fields: Text
    message_sources: Text
    turns_with_history: Text
    last_turn_not_customer: Text
    duplicate_case_id: Text
    no_rows: Text
    list_separator: RawText
    line_error: Text
    field_error: Text
    field_error_separator: RawText
    location_separator: RawText
    row_location: Text
    error_line: Text
    at_least_one: Text
    unknown_personas: Text
    unknown_label: Text
    missing_detection_reference: Text
    missing_simulation_reference: Text
    judge_failed: Text


class PersonaSettings(_Section):
    """One persona: whether it attacks, and its own part of the system prompt."""

    kind: PersonaKind
    prompt: Text


class JailbreakEvalSettings(BaseSettings):
    """All jailbreak eval settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH,
        env_prefix="JAILBREAK_EVALS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    paths: PathSettings
    cli: CliSettings
    help: HelpSettings
    langsmith: LangSmithSettings
    data: DataSettings
    channels: ChannelSettings
    detection: DetectionSettings
    simulation: SimulationSettings
    models: ModelSettings
    texts: TextSettings
    personas: dict[Text, PersonaSettings] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        """Reject a key or chat role that two settings share where code needs one."""
        simulation = self.simulation
        groups = {
            "data keys": (*self.data.row_keys, *self.data.reserved_keys),
            "detection feedback keys": self.detection.feedback_keys,
            "simulation feedback keys": simulation.feedback_keys,
            "cli metadata keys": tuple(self.cli.metadata_keys.model_dump().values()),
            "simulation roles": (
                simulation.customer_role,
                simulation.agent_role,
                simulation.system_role,
            ),
        }
        for group, names in groups.items():
            repeated = _repeated(names)
            if repeated:
                raise ValueError(f"{group} must be unique, but repeat {repeated}")
        return self

    @model_validator(mode="after")
    def _check_templates(self) -> Self:
        """Reject a template whose placeholders differ from those the code fills in."""
        for key, (template, names) in self._templates().items():
            if _placeholders(template) != names:
                raise ValueError(
                    f"{key} must use exactly the placeholders {sorted(names)}"
                )
        return self

    def _templates(self) -> dict[str, tuple[str, set[str]]]:
        """Return each formatted text by its key, with the placeholders code fills."""
        texts, help_texts = self.texts, self.help
        return {
            "texts.no_langsmith_key": (texts.no_langsmith_key, {"env"}),
            "texts.no_openai_key": (texts.no_openai_key, {"env"}),
            "texts.no_data": (texts.no_data, {"data_dir"}),
            "texts.data_not_found": (texts.data_not_found, {"paths"}),
            "texts.not_positive_int": (texts.not_positive_int, {"text"}),
            "texts.reserved_fields": (texts.reserved_fields, {"fields"}),
            "texts.message_sources": (texts.message_sources, {"keys"}),
            "texts.turns_with_history": (
                texts.turns_with_history,
                {"turns", "history"},
            ),
            "texts.last_turn_not_customer": (texts.last_turn_not_customer, {"turns"}),
            "texts.duplicate_case_id": (texts.duplicate_case_id, {"case_id"}),
            "texts.no_rows": (texts.no_rows, {"path"}),
            "texts.line_error": (texts.line_error, {"path", "line", "message"}),
            "texts.field_error": (texts.field_error, {"location", "reason"}),
            "texts.error_line": (texts.error_line, {"prog", "error"}),
            "texts.at_least_one": (texts.at_least_one, {"name", "value"}),
            "texts.unknown_personas": (texts.unknown_personas, {"unknown", "known"}),
            "texts.unknown_label": (texts.unknown_label, {"label"}),
            "simulation.turn_budget": (self.simulation.turn_budget, {"follow_ups"}),
            "langsmith.report_line": (self.langsmith.report_line, {"label", "value"}),
            "help.detection_data": (help_texts.detection_data, {"data_dir"}),
            "help.simulation_data": (help_texts.simulation_data, {"data_dir", "file"}),
            "help.personas": (help_texts.personas, {"names"}),
        }

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Read init arguments, then the environment, then .env, then config.toml."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )


settings = JailbreakEvalSettings()
