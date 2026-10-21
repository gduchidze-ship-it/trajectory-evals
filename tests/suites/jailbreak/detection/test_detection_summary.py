"""ConfusionMatrix edge cases and the summary evaluator that reports it."""

from collections.abc import Mapping
from typing import Any, cast

import pytest

from trajectory_evals.suites.jailbreak.detection import (
    ConfusionMatrix,
    create_detection_summary_evaluators,
)
from trajectory_evals.suites.jailbreak.detection import summary as summary_module
from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.types import DetectionOutputs, DetectionReference, ExpectedAction
from tests.suites.jailbreak.detection.detection_fakes import (
    detection_outputs,
    errored_outputs,
    unknown_label_reference,
)

Row = tuple[ExpectedAction, bool, str]
MIXED_ROWS: list[Row] = [
    ("BLOCKED", True, "judge"),
    ("BLOCKED", False, "screen"),
    ("BLOCKED", False, "fail_policy"),
    ("ALLOWED", True, "judge"),
    ("ALLOWED", False, "screen"),
    ("ALLOWED", False, "screen"),
    ("ALLOWED", False, "judge"),
    ("REVIEW", True, "judge"),
]
MIXED_METRICS = {
    "false_positive_rate": 1 / 4,
    "detection_rate": 1 / 3,
    "precision": 1 / 2,
    "f1": 2 / 5,
    "escalation_rate": 5 / 8,
    "fail_open_rate": 1 / 8,
    "review_block_rate": 1.0,
    "error_rate": 0.0,
}


def _runs(
    rows: list[Row],
) -> tuple[list[DetectionOutputs], list[DetectionReference]]:
    """Split (label, verdict, decided_by) rows into outputs and reference outputs."""
    outputs = [
        detection_outputs(blocked, decided_by) for _, blocked, decided_by in rows
    ]
    references = [DetectionReference(expected_action=label) for label, _, _ in rows]
    return outputs, references


def _metrics(matrix: ConfusionMatrix) -> dict[str, float | None]:
    """Return every rate the summary evaluators report, by metric name."""
    return {
        "false_positive_rate": matrix.false_positive_rate,
        "detection_rate": matrix.detection_rate,
        "precision": matrix.precision,
        "f1": matrix.f1,
        "escalation_rate": matrix.escalation_rate,
        "fail_open_rate": matrix.fail_open_rate,
        "review_block_rate": matrix.review_block_rate,
        "error_rate": matrix.error_rate,
    }


def _summary(metrics: Mapping[str, float | None]) -> dict[str, list[dict[str, Any]]]:
    """Return the batch feedback the summary evaluator reports for these metrics."""
    return {"results": [{"key": key, "score": score} for key, score in metrics.items()]}


def test_mixed_rows_fill_every_cell() -> None:
    """Each row lands in one cell, and routes are counted across every label."""
    matrix = ConfusionMatrix.from_runs(*_runs(MIXED_ROWS))
    assert matrix == ConfusionMatrix(
        tp=1, fp=1, tn=3, fn=2, review_blocked=1, escalated=5, failed_open=1
    )
    assert (matrix.total, matrix.review_total) == (8, 1)
    assert _metrics(matrix) == MIXED_METRICS


def test_an_errored_run_counts_only_toward_error_rate() -> None:
    """A run whose target raised is skipped, and every metric is still reported."""
    outputs, references = _runs(MIXED_ROWS)
    outputs.append(errored_outputs())
    references.append(DetectionReference(expected_action="BLOCKED"))
    matrix = ConfusionMatrix.from_runs(outputs, references)
    assert matrix == ConfusionMatrix(
        tp=1,
        fp=1,
        tn=3,
        fn=2,
        review_blocked=1,
        escalated=5,
        failed_open=1,
        errored=1,
    )
    assert (matrix.answered, matrix.total) == (8, 9)
    [summary] = create_detection_summary_evaluators()
    assert summary(outputs=outputs, reference_outputs=references) == _summary(
        {**MIXED_METRICS, "error_rate": 1 / 9}
    )


def test_a_benign_only_file_has_no_detection_metrics() -> None:
    """Without a BLOCKED row, detection rate, precision and F1 are undefined."""
    rows: list[Row] = [
        ("ALLOWED", True, "judge"),
        ("ALLOWED", False, "judge"),
        ("ALLOWED", False, "screen"),
        ("ALLOWED", False, "screen"),
    ]
    assert _metrics(ConfusionMatrix.from_runs(*_runs(rows))) == {
        "false_positive_rate": 1 / 4,
        "detection_rate": None,
        "precision": None,
        "f1": None,
        "escalation_rate": 1 / 2,
        "fail_open_rate": 0.0,
        "review_block_rate": None,
        "error_rate": 0.0,
    }


def test_no_rows_leave_every_metric_undefined() -> None:
    """With every denominator 0, every metric is None."""
    matrix = ConfusionMatrix.from_runs([], [])
    assert matrix == ConfusionMatrix()
    assert set(_metrics(matrix).values()) == {None}


def test_review_rows_count_only_toward_review_and_routes() -> None:
    """REVIEW rows never enter the confusion cells."""
    rows: list[Row] = [("REVIEW", True, "judge"), ("REVIEW", False, "screen")]
    assert _metrics(ConfusionMatrix.from_runs(*_runs(rows))) == {
        "false_positive_rate": None,
        "detection_rate": None,
        "precision": None,
        "f1": None,
        "escalation_rate": 1 / 2,
        "fail_open_rate": 0.0,
        "review_block_rate": 1 / 2,
        "error_rate": 0.0,
    }


def test_missing_every_attack_scores_f1_zero() -> None:
    """A detector that blocks nothing on attacks scores 0, though precision is None."""
    matrix = ConfusionMatrix.from_runs(*_runs([("BLOCKED", False, "screen")]))
    assert (matrix.detection_rate, matrix.precision, matrix.f1) == (0.0, None, 0.0)


def test_an_unknown_label_is_rejected() -> None:
    """A label outside ExpectedAction raises, naming the label."""
    outputs = [detection_outputs(False, "screen")]
    with pytest.raises(ValueError, match="UNSURE"):
        ConfusionMatrix.from_runs(outputs, [unknown_label_reference()])


def test_a_malformed_output_fails_loudly() -> None:
    """An answer missing its route is a target bug, not an errored run."""
    malformed = cast(DetectionOutputs, {"is_jailbreak": True})
    with pytest.raises(KeyError):
        ConfusionMatrix.from_runs([malformed], [{"expected_action": "BLOCKED"}])


def test_one_summary_evaluator_reports_every_metric() -> None:
    """A single evaluator reports each metric as its own feedback key."""
    outputs, references = _runs(MIXED_ROWS)
    [summary] = create_detection_summary_evaluators()
    assert summary(outputs=outputs, reference_outputs=references) == _summary(
        MIXED_METRICS
    )


def test_the_summary_scores_undefined_metrics_none() -> None:
    """An undefined metric is reported with a None score rather than left out."""
    [summary] = create_detection_summary_evaluators()
    assert summary(outputs=[], reference_outputs=[]) == _summary(
        dict.fromkeys(MIXED_METRICS)
    )


def test_a_renamed_metric_key_is_reported_under_its_new_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each metric is reported under the key [detection.metrics] gives it."""
    metrics = settings.detection.metrics.model_copy(update={"f1": "f1_score"})
    detection = settings.detection.model_copy(update={"metrics": metrics})
    monkeypatch.setattr(
        summary_module, "settings", settings.model_copy(update={"detection": detection})
    )
    [summary] = create_detection_summary_evaluators()

    keys = [
        result["key"] for result in summary(outputs=[], reference_outputs=[])["results"]
    ]

    assert keys == [key if key != "f1" else "f1_score" for key in MIXED_METRICS]
