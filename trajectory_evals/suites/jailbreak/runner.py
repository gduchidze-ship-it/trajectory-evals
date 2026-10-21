"""Jailbreak experiment metadata, beside the core experiment runner it re-exports."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from src.toolsets.jailbreak_detection.settings import JailbreakDetectionSettings
from src.toolsets.jailbreak_detection.settings import settings as shipped_settings

from trajectory_evals.core.runner import (
    ExperimentConfig,
    ExperimentReport,
    ExperimentRunner,
    Target,
)
from trajectory_evals.suites.jailbreak.settings import settings as eval_settings


def _hash_prompts(settings: JailbreakDetectionSettings) -> str:
    """Return a sha256 over both prompts, every tool text and every finding text."""
    texts = (
        settings.screen_prompt,
        settings.judge_prompt,
        settings.tools.model_dump_json(),
        settings.findings.model_dump_json(),
    )
    separator = eval_settings.langsmith.prompt_hash_separator
    return hashlib.sha256(separator.join(texts).encode()).hexdigest()


@dataclass(frozen=True)
class ExperimentMetadata:
    """The models, guardrail, prompts and dataset a jailbreak experiment ran against."""

    screen_model: str
    judge_model: str
    guardrail_id: str
    guardrail_version: str
    prompts_sha256: str
    dataset_file: str
    row_count: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def collect(
        cls,
        *,
        dataset_file: Path,
        row_count: int,
        extra: Mapping[str, Any] | None = None,
        settings: JailbreakDetectionSettings = shipped_settings,
    ) -> Self:
        """Gather one dataset file's metadata from the detection settings."""
        return cls(
            screen_model=settings.models.screen,
            judge_model=settings.models.judge,
            guardrail_id=settings.guardrail.id,
            guardrail_version=settings.guardrail.version,
            prompts_sha256=_hash_prompts(settings),
            dataset_file=dataset_file.name,
            row_count=row_count,
            extra=dict(extra or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the fields as flat experiment metadata, extra keys first."""
        recorded = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "extra"
        }
        return {**self.extra, **recorded}


__all__ = [
    "ExperimentConfig",
    "ExperimentMetadata",
    "ExperimentReport",
    "ExperimentRunner",
    "Target",
]
