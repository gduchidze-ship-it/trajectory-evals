"""Settings building blocks every package reuses, and core's own settings."""

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from string import Formatter
from typing import Annotated, Literal, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
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


def _require_unique(groups: Mapping[str, Iterable[str]]) -> None:
    """Raise ValueError naming the first group whose names repeat."""
    for group, names in groups.items():
        repeated = _repeated(names)
        if repeated:
            raise ValueError(f"{group} must be unique, but repeat {repeated}")


def _require_placeholders(templates: Mapping[str, tuple[str, set[str]]]) -> None:
    """Raise ValueError naming the first template whose placeholders differ."""
    for key, (template, names) in templates.items():
        if _placeholders(template) != names:
            raise ValueError(f"{key} must use exactly the placeholders {sorted(names)}")


class _Section(BaseModel):
    """A frozen config.toml table that rejects unknown keys, so a typo fails."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PackageSettings(BaseSettings):
    """A package's settings; subclasses set only toml_file and env_prefix."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    @classmethod
    @override
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


class EvaluatorSettings(_Section):
    """The outputs key LangSmith sets to None when a target raised."""

    errored_output_key: Literal["output"]


class TextSettings(_Section):
    """Messages for a bad data row, a bad agent folder or a misused API."""

    not_found: Text
    file_error: Text
    no_endpoint: Text
    outside_root: Text
    not_an_object: Text
    duplicate_case_id: Text
    line_error: Text
    field_error: Text
    field_error_separator: RawText
    location_separator: RawText
    row_location: Text
    at_least_one: Text


class ReportLabels(_Section):
    """The label of each fixed line of the printed experiment report."""

    experiment: Text
    dataset: Text
    url: Text
    runs: Text


class LangSmithSettings(_Section):
    """Experiment defaults, example fetch batches and the printed report."""

    repetitions: PositiveInt
    ids_per_request: PositiveInt
    local_suffix: RawText
    not_uploaded: Text
    undefined_score: Text
    score_decimals: NonNegativeInt
    report_line: Text
    report_labels: ReportLabels


class DataSettings(_Section):
    """The example metadata keys the dataset mirror writes, and its write batches."""

    case_id_key: Text
    content_hash_key: Text
    update_batch_size: PositiveInt
    delete_batch_size: PositiveInt


class CoreSettings(PackageSettings):
    """Core settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH, env_prefix="TRAJECTORY_EVALS_CORE_"
    )

    evaluators: EvaluatorSettings
    texts: TextSettings
    langsmith: LangSmithSettings
    data: DataSettings

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        """Reject example metadata keys that would overwrite each other."""
        data = self.data
        _require_unique({"data keys": (data.case_id_key, data.content_hash_key)})
        return self

    @model_validator(mode="after")
    def _check_templates(self) -> Self:
        """Reject a template whose placeholders differ from those the code fills in."""
        texts = self.texts
        _require_placeholders(
            {
                "texts.not_found": (texts.not_found, {"path"}),
                "texts.file_error": (texts.file_error, {"path", "message"}),
                "texts.no_endpoint": (texts.no_endpoint, {"path"}),
                "texts.outside_root": (texts.outside_root, {"path", "root"}),
                "texts.duplicate_case_id": (texts.duplicate_case_id, {"case_id"}),
                "texts.line_error": (texts.line_error, {"path", "line", "message"}),
                "texts.field_error": (texts.field_error, {"location", "reason"}),
                "texts.at_least_one": (texts.at_least_one, {"name", "value"}),
                "langsmith.report_line": (
                    self.langsmith.report_line,
                    {"label", "value"},
                ),
            }
        )
        return self


settings = CoreSettings()

__all__ = [
    "CONFIG_PATH",
    "PACKAGE_DIR",
    "REPO_ROOT",
    "CoreSettings",
    "DataSettings",
    "EvaluatorSettings",
    "LangSmithSettings",
    "PackageSettings",
    "RawText",
    "ReportLabels",
    "Text",
    "TextSettings",
    "_Section",
    "_require_placeholders",
    "_require_unique",
    "settings",
]
