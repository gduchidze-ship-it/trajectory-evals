"""Tests for writing an experiment's rows, summary and metadata to disk."""

import dataclasses
import json
import re
from pathlib import Path

import pytest

from tests.core.core_fakes import make_row
from trajectory_evals.core.artifacts import write_run
from trajectory_evals.core.runner import ExperimentReport

_SUMMARY = {"pass_rate": 0.5, "pass_rate_ci_low": None}
_META = {"agent": "support", "prompts": {"system": "sha256:abc"}}


def _report(experiment_name: str = "golden (local)") -> ExperimentReport:
    """Return a report of two rows, one holding non-ASCII text."""
    rows = (
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

    lines = (run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert run_dir == tmp_path / "results" / "run-1"
    assert [json.loads(line) for line in lines] == [
        dataclasses.asdict(row) for row in report.rows
    ]
    assert json.loads((run_dir / "summary.json").read_text()) == _SUMMARY
    assert json.loads((run_dir / "meta.json").read_text()) == _META


def test_write_run_refuses_to_overwrite_a_run(tmp_path: Path) -> None:
    """A second run under the same id fails and leaves the first one as it was."""
    run_dir = write_run(
        _report(), results_dir=tmp_path, summary=_SUMMARY, meta=_META, run_id="run-1"
    )
    first = (run_dir / "summary.json").read_text()

    with pytest.raises(FileExistsError):
        write_run(_report(), results_dir=tmp_path, summary={}, meta={}, run_id="run-1")

    assert (run_dir / "summary.json").read_text() == first


def test_default_run_id_is_the_utc_time_and_a_slug(tmp_path: Path) -> None:
    """Without a run id, the directory is named by UTC time and experiment slug."""
    run_dir = write_run(
        _report("Detect Jailbreak (local)"),
        results_dir=tmp_path,
        summary=_SUMMARY,
        meta=_META,
    )

    assert re.fullmatch(r"\d{8}_\d{6}_detect-jailbreak-local", run_dir.name)
