"""Tests for writing an experiment's rows, summary and metadata to disk."""

import dataclasses
import json
import re
from pathlib import Path

import pytest

from tests.core.core_fakes import make_row
from trajectory_evals.core.artifacts import write_run
from trajectory_evals.core.runner import ExperimentReport, ExperimentRow

_SUMMARY = {"pass_rate": 0.5, "pass_rate_ci_low": None}
_META = {"agent": "support", "prompts": {"system": "sha256:abc"}}


class _Unprintable:
    """A value that JSON cannot hold and that fails to turn into a string."""

    def __str__(self) -> str:
        """Fail, as a broken custom object would."""
        raise RuntimeError("cannot print")


def _report(
    experiment_name: str = "golden (local)", *rows: ExperimentRow
) -> ExperimentReport:
    """Return a report of rows, by default two with one holding non-ASCII text."""
    rows = rows or (
        make_row("a", {"ok": True}, inputs={"text": "Grüße"}, latency_ms=12.5),
        make_row("b", {"ok": None}, error="RuntimeError('boom')"),
    )
    return ExperimentReport(
        experiment_name=experiment_name,
        dataset_name="golden/sample",
        url=None,
        summary={},
        runs=len(rows),
        rows=rows,
    )


def _read_rows(run_dir: Path) -> list[dict[str, object]]:
    """Return every object in the run's rows.jsonl, in order."""
    lines = (run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_write_run_round_trips_rows_summary_and_meta(tmp_path: Path) -> None:
    """Each row is one JSON line, and summary and meta read back as written."""
    report = _report()

    run_dir = write_run(
        report,
        results_dir=tmp_path / "results",
        summary=_SUMMARY,
        meta=_META,
        run_id="run-1",
    )

    assert run_dir == tmp_path / "results" / "run-1"
    assert _read_rows(run_dir) == [
        {**dataclasses.asdict(row), "feedback_errors": []} for row in report.rows
    ]
    assert json.loads((run_dir / "summary.json").read_text()) == _SUMMARY
    assert json.loads((run_dir / "meta.json").read_text()) == _META


def test_write_run_writes_values_json_cannot_hold(tmp_path: Path) -> None:
    """Keys become str, sets sorted lists, and NaN or infinity becomes null."""
    row = make_row(
        "a",
        outputs={1: "one", "score": float("nan")},
        feedback_errors=frozenset({"wrapper", "judge"}),
    )

    run_dir = write_run(
        _report("golden (local)", row),
        results_dir=tmp_path,
        summary={"pass_rate": float("inf")},
        meta={},
        run_id="run-1",
    )

    [written] = _read_rows(run_dir)
    assert written["outputs"] == {"1": "one", "score": None}
    assert written["feedback_errors"] == ["judge", "wrapper"]
    assert json.loads((run_dir / "summary.json").read_text()) == {"pass_rate": None}


def test_write_run_makes_no_directory_when_a_file_fails(tmp_path: Path) -> None:
    """Every file is serialized first, so a value that fails leaves no run behind."""
    with pytest.raises(RuntimeError, match="cannot print"):
        write_run(
            _report(),
            results_dir=tmp_path,
            summary=_SUMMARY,
            meta={"bad": _Unprintable()},
            run_id="run-1",
        )

    assert list(tmp_path.iterdir()) == []


def test_write_run_refuses_to_overwrite_a_run(tmp_path: Path) -> None:
    """A second run under the same id fails and leaves the first one as it was."""
    run_dir = write_run(
        _report(), results_dir=tmp_path, summary=_SUMMARY, meta=_META, run_id="run-1"
    )
    first = (run_dir / "summary.json").read_text()

    with pytest.raises(FileExistsError):
        write_run(_report(), results_dir=tmp_path, summary={}, meta={}, run_id="run-1")

    assert (run_dir / "summary.json").read_text() == first


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "/tmp/run", ".", ".."])
def test_write_run_refuses_a_run_id_that_is_not_one_name(
    tmp_path: Path, run_id: str
) -> None:
    """A run id that names a path, or . or .., is refused before anything is made."""
    results_dir = tmp_path / "results"

    with pytest.raises(ValueError, match="one directory name"):
        write_run(
            _report(), results_dir=results_dir, summary={}, meta={}, run_id=run_id
        )

    assert list(tmp_path.iterdir()) == []


def test_default_run_ids_are_time_slug_and_unique(tmp_path: Path) -> None:
    """Without a run id, two runs in the same second still get distinct folders."""
    report = _report("Detect Jailbreak (local)")

    first, second = (
        write_run(report, results_dir=tmp_path, summary=_SUMMARY, meta=_META)
        for _ in range(2)
    )

    pattern = r"\d{8}_\d{6}_detect-jailbreak-local_[0-9a-f]{6}"
    assert re.fullmatch(pattern, first.name)
    assert re.fullmatch(pattern, second.name)
    assert first != second
