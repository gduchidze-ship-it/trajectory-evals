"""Golden eval settings: config.toml values, overridable by environment."""

from pathlib import Path
from typing import Self

from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

from trajectory_evals.core.settings import (
    PackageSettings,
    RawText,
    Text,
    _require_placeholders,
    _require_unique,
    _Section,
)

PACKAGE_DIR = Path(__file__).parent
CONFIG_PATH = PACKAGE_DIR / "config.toml"


class LangSmithSettings(_Section):
    """The framework every grader's trace is tagged with."""

    ls_framework: Text


class FeedbackKeys(_Section):
    """The feedback key each built-in grader reports under."""

    output_exact: Text
    contains: Text
    not_contains: Text
    tools_called: Text
    trajectory_match: Text
    criteria: Text


class DataSettings(_Section):
    """Chat roles graders read, and the metadata keys the data layer writes."""

    user_role: Text
    assistant_role: Text
    source_file_key: Text
    reserved_keys: tuple[Text, ...]


class CommentSettings(_Section):
    """The short reason a grader gives; a reflection model reads the failures."""

    target_errored: Text
    not_expected: Text
    trajectory_unreported: Text
    output_mismatch: Text
    missing: Text
    forbidden: Text
    tools_missing: Text
    trajectory_mismatch: Text
    judge_error: Text
    item_separator: RawText


class JudgeSettings(_Section):
    """The default judge model, the openevals prompt prefix and the criteria prompt."""

    model: Text
    openevals_prefix: Text
    criteria_prompt: Text


class TextSettings(_Section):
    """Messages for a bad golden row, or a bad [judges] or [graders] table."""

    input_or_messages: Text
    blank_input: Text
    last_not_user: Text
    reserved_fields: Text
    list_separator: RawText
    table_error: Text
    bad_placeholders: Text
    unknown_prompt: Text
    reference_free_prompt: Text
    continuous_gate: Text
    key_taken: Text
    not_graders: Text
    judge_failed: Text


class GoldenSettings(PackageSettings):
    """All golden eval settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH, env_prefix="TRAJECTORY_EVALS_GOLDEN_"
    )

    langsmith: LangSmithSettings
    keys: FeedbackKeys
    data: DataSettings
    comments: CommentSettings
    judges: JudgeSettings
    texts: TextSettings

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        """Reject a feedback key two graders share, or a role used twice."""
        data = self.data
        _require_unique(
            {
                "feedback keys": self.keys.model_dump().values(),
                "chat roles": (data.user_role, data.assistant_role),
                "metadata keys": (data.source_file_key, *data.reserved_keys),
            }
        )
        return self

    @model_validator(mode="after")
    def _check_templates(self) -> Self:
        """Reject a template whose placeholders differ from those the code fills in."""
        comments, texts = self.comments, self.texts
        _require_placeholders(
            {
                "comments.not_expected": (comments.not_expected, {"key"}),
                "comments.output_mismatch": (comments.output_mismatch, {"expected"}),
                "comments.missing": (comments.missing, {"items"}),
                "comments.forbidden": (comments.forbidden, {"items"}),
                "comments.tools_missing": (comments.tools_missing, {"items"}),
                "comments.trajectory_mismatch": (
                    comments.trajectory_mismatch,
                    {"trajectory_match", "tool_args_match"},
                ),
                "judges.criteria_prompt": (
                    self.judges.criteria_prompt,
                    {"context", "transcript", "criteria"},
                ),
                "texts.reserved_fields": (texts.reserved_fields, {"fields"}),
                "texts.table_error": (texts.table_error, {"path", "table", "message"}),
                "texts.bad_placeholders": (
                    texts.bad_placeholders,
                    {"allowed", "used"},
                ),
                "texts.unknown_prompt": (texts.unknown_prompt, {"name"}),
                "texts.key_taken": (texts.key_taken, {"key"}),
                "texts.not_graders": (texts.not_graders, {"factory"}),
            }
        )
        return self


settings = GoldenSettings()

__all__ = [
    "CONFIG_PATH",
    "PACKAGE_DIR",
    "CommentSettings",
    "DataSettings",
    "FeedbackKeys",
    "GoldenSettings",
    "JudgeSettings",
    "LangSmithSettings",
    "TextSettings",
    "settings",
]
