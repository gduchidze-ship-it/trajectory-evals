"""Jailbreak example settings: config.toml values, overridable by environment."""

from pathlib import Path
from typing import Self

from pydantic import Field, PositiveInt, model_validator
from pydantic_settings import SettingsConfigDict

from trajectory_evals.core.settings import (
    PackageSettings,
    RawText,
    Text,
    _require_placeholders,
    _Section,
)

PACKAGE_DIR = Path(__file__).parent
CONFIG_PATH = PACKAGE_DIR / "config.toml"


class KeySettings(_Section):
    """The tool name, the golden label key, each row feedback key and the trace tag."""

    ls_framework: Text
    tool: Text
    action: Text
    correct: Text
    escalated: Text
    failed_open: Text
    simulation_correct: Text
    turns_to_block: Text
    attack_attempted: Text
    persona_faithful: Text


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


class SimulationMetricSettings(_Section):
    """The key each simulation summary measure is reported under, in report order."""

    attack_blocked: Text
    benign_blocked: Text
    attack_turns_to_block: Text
    faithful: Text
    errored: Text


class MetricSettings(_Section):
    """The summary keys of the detection and simulation graders."""

    detection: DetectionMetricSettings
    simulation: SimulationMetricSettings


class ResponderSettings(_Section):
    """The tool call id, the override the assistant applies, its model and region."""

    tool_call_id: Text
    system_override: Text
    model: Text
    agent_initiator: Text
    aws_region_env: Text
    default_aws_region: Text


class ChannelSettings(_Section):
    """The context key holding a message's channel, and the channels told apart."""

    key: Text
    sms: Text
    conversation_only: Text
    opener_text_only: Text


class ScenarioSettings(_Section):
    """The agent whose personas scenarios use, the sample and the two registers."""

    agent: Text
    sample: PositiveInt
    seed: int
    unknown_channel: Text
    sms_register: Text
    email_register: Text


class DataSettings(_Section):
    """The prod export's row keys; the converter keeps every other key as metadata."""

    id_key: Text
    label_key: Text
    text_keys: tuple[Text, ...] = Field(min_length=1)
    turns_key: Text
    history_key: Text

    @property
    def message_keys(self) -> tuple[str, ...]:
        """Return the row keys that can hold the message, the turns key last."""
        return (*self.text_keys, self.turns_key)

    @property
    def row_keys(self) -> tuple[str, ...]:
        """Return every row key a case reads; the other keys become metadata."""
        return (self.id_key, self.label_key, self.history_key, *self.message_keys)


class ConvertSettings(_Section):
    """The converter's name, help texts and exit code on bad input."""

    prog: Text
    description: Text
    input_help: Text
    output_help: Text
    scenarios_help: Text
    sample_help: Text
    seed_help: Text
    allow_tracked_help: Text
    input_error_exit_code: PositiveInt


class CommentSettings(_Section):
    """Why a grader left a row unscored."""

    target_errored: Text
    review: Text
    opener_blocked: Text
    benign: Text
    not_blocked: Text
    not_judged: Text
    not_attempted: Text


class TextSettings(_Section):
    """Messages for bad input, a missing prerequisite or a misused API."""

    no_aws_credentials: Text
    reply_needs_prompts: Text
    missing_action: Text
    unknown_label: Text
    no_result: Text
    judge_failed: Text
    message_sources: Text
    turns_with_history: Text
    last_turn_not_customer: Text
    no_message: Text
    reserved_fields: Text
    list_separator: RawText
    error_line: Text
    converted: Text


class JailbreakExampleSettings(PackageSettings):
    """All jailbreak example settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH, env_prefix="JAILBREAK_EXAMPLE_"
    )

    keys: KeySettings
    metrics: MetricSettings
    responder: ResponderSettings
    channels: ChannelSettings
    scenarios: ScenarioSettings
    data: DataSettings
    convert: ConvertSettings
    comments: CommentSettings
    texts: TextSettings

    @model_validator(mode="after")
    def _check_templates(self) -> Self:
        """Reject a template whose placeholders differ from those the code fills in."""
        texts = self.texts
        _require_placeholders(
            {
                "responder.tool_call_id": (self.responder.tool_call_id, {"turn"}),
                "texts.missing_action": (texts.missing_action, {"key"}),
                "texts.unknown_label": (texts.unknown_label, {"label"}),
                "texts.no_result": (texts.no_result, {"tool"}),
                "texts.message_sources": (texts.message_sources, {"keys"}),
                "texts.turns_with_history": (
                    texts.turns_with_history,
                    {"turns", "history"},
                ),
                "texts.last_turn_not_customer": (
                    texts.last_turn_not_customer,
                    {"turns"},
                ),
                "texts.reserved_fields": (texts.reserved_fields, {"fields"}),
                "texts.error_line": (texts.error_line, {"prog", "error"}),
                "texts.converted": (texts.converted, {"count", "path"}),
            }
        )
        return self


settings = JailbreakExampleSettings()

__all__ = [
    "CONFIG_PATH",
    "PACKAGE_DIR",
    "JailbreakExampleSettings",
    "settings",
]
