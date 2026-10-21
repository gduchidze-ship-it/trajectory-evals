"""Tests for the jailbreak experiment metadata the runner module collects."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from trajectory_evals.suites.jailbreak.runner import ExperimentMetadata
from src.toolsets.jailbreak_detection.settings import (
    FindingSettings,
    JailbreakDetectionSettings,
    ToolSettings,
    settings,
)

_DATASET_FILE = Path("data/jailbreak/sample.jsonl")
_PROMPTS = {
    "screen_prompt": "screen prompt",
    "judge_prompt": "judge prompt",
    "reminder": "report it",
    "description": "report the verdict",
    "reason": "say why",
    "screen_failed": "screen failed",
}
PINNED_PROMPTS_SHA256 = (
    "66b24fb6c13bbdaf570abc1068a07624eb7d650cf183a7e2f0e646f104505174"
)


def _fake_settings(
    screen_prompt: str,
    judge_prompt: str,
    reminder: str,
    description: str,
    reason: str,
    screen_failed: str,
) -> JailbreakDetectionSettings:
    """Return settings carrying fixed models and guardrail plus the given texts."""
    tools = ToolSettings(
        report_verdict_name="report_verdict",
        report_verdict_description=description,
        report_verdict_reason_description=reason,
        report_verdict_reminder=reminder,
        read_history_name="read_history",
        read_history_description="read the history",
    )
    findings = FindingSettings(
        reason_max_length=50,
        blocked_detection="{policy} {detections} {fields}",
        guardrail_timed_out="guardrail timed out",
        guardrail_failed="guardrail failed",
        screen_timed_out="screen timed out",
        screen_failed=screen_failed,
    )
    return cast(
        JailbreakDetectionSettings,
        SimpleNamespace(
            models=SimpleNamespace(screen="screen-model", judge="judge-model"),
            guardrail=SimpleNamespace(id="guardrail-1", version="7"),
            screen_prompt=screen_prompt,
            judge_prompt=judge_prompt,
            tools=tools,
            findings=findings,
        ),
    )


def _prompts_hash(**prompts: str) -> str:
    """Return the prompt hash collected from fake settings with the given prompts."""
    return ExperimentMetadata.collect(
        dataset_file=_DATASET_FILE,
        row_count=1,
        settings=_fake_settings(**prompts),
    ).prompts_sha256


def test_collect_records_the_settings_dataset_and_extra_keys() -> None:
    """Collected metadata flattens the models, guardrail, prompts and dataset."""
    metadata = ExperimentMetadata.collect(
        dataset_file=_DATASET_FILE,
        row_count=3,
        extra={"suite": "detection"},
        settings=_fake_settings(**_PROMPTS),
    )

    assert metadata.as_dict() == {
        "suite": "detection",
        "screen_model": "screen-model",
        "judge_model": "judge-model",
        "guardrail_id": "guardrail-1",
        "guardrail_version": "7",
        "prompts_sha256": _prompts_hash(**_PROMPTS),
        "dataset_file": "sample.jsonl",
        "row_count": 3,
    }


def test_collect_defaults_to_the_shipped_settings() -> None:
    """Without settings, collect reads the shipped models and guardrail."""
    metadata = ExperimentMetadata.collect(dataset_file=_DATASET_FILE, row_count=1)

    assert (
        metadata.screen_model,
        metadata.judge_model,
        metadata.guardrail_id,
        metadata.guardrail_version,
    ) == (
        settings.models.screen,
        settings.models.judge,
        settings.guardrail.id,
        settings.guardrail.version,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"screen_prompt": "screen prompt v2"},
        {"judge_prompt": "judge prompt v2"},
        {"reminder": "report it now"},
        {"description": "report the verdict now"},
        {"reason": "say why now"},
        {"screen_failed": "screen broke"},
        {"screen_prompt": "screen", "judge_prompt": " promptjudge prompt"},
    ],
)
def test_prompt_hash_changes_with_any_prompt_text(change: dict[str, Any]) -> None:
    """Editing a prompt, tool or finding text, or moving text across, changes it."""
    assert _prompts_hash(**{**_PROMPTS, **change}) != _prompts_hash(**_PROMPTS)


def test_prompt_hash_is_pinned() -> None:
    """The hash covers both prompts and the tool and finding JSON, NUL-separated."""
    assert _prompts_hash(**_PROMPTS) == PINNED_PROMPTS_SHA256
