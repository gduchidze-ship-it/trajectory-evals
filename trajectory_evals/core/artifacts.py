"""One experiment's rows, summary and run metadata, written under results/<run_id>/."""

import dataclasses
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trajectory_evals.core.runner import ExperimentReport

_ROWS_FILE = "rows.jsonl"
_SUMMARY_FILE = "summary.json"
_META_FILE = "meta.json"
_RUN_ID_TIME = "%Y%m%d_%H%M%S"
_NOT_SLUG = re.compile(r"[^a-z0-9]+")
_INDENT = 2


def write_run(
    report: ExperimentReport,
    *,
    results_dir: Path,
    summary: Mapping[str, Any],
    meta: Mapping[str, Any],
    run_id: str | None = None,
) -> Path:
    """Write the run to results_dir/<run_id>/ and return that directory.

    rows.jsonl holds one JSON object per report row, beside summary.json and
    meta.json; a value JSON cannot hold is written as its str(). run_id defaults
    to the UTC time plus a slug of the experiment name. An existing run
    directory raises FileExistsError, so no run is overwritten. The rows hold
    customer text, and results/ is gitignored.
    """
    run_dir = results_dir / (run_id or _default_run_id(report.experiment_name))
    run_dir.mkdir(parents=True)
    rows = "".join(_to_json(dataclasses.asdict(row)) for row in report.rows)
    (run_dir / _ROWS_FILE).write_text(rows, encoding="utf-8")
    (run_dir / _SUMMARY_FILE).write_text(_to_json(summary, _INDENT), encoding="utf-8")
    (run_dir / _META_FILE).write_text(_to_json(meta, _INDENT), encoding="utf-8")
    return run_dir


def _default_run_id(experiment_name: str) -> str:
    """Return the current UTC time and a lowercase slug of the experiment name."""
    slug = _NOT_SLUG.sub("-", experiment_name.lower()).strip("-")
    return f"{datetime.now(UTC).strftime(_RUN_ID_TIME)}_{slug}"


def _to_json(value: Mapping[str, Any], indent: int | None = None) -> str:
    """Return value as UTF-8 JSON ending with a newline, on one line unless indented."""
    text = json.dumps(dict(value), indent=indent, default=str, ensure_ascii=False)
    return text + "\n"


__all__ = ["write_run"]
