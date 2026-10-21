"""One experiment's rows, summary and run metadata, written under results/<run_id>/."""

import json
import math
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trajectory_evals.core.runner import ExperimentReport
from trajectory_evals.core.settings import settings

_ROWS_FILE = "rows.jsonl"
_SUMMARY_FILE = "summary.json"
_META_FILE = "meta.json"
_RUN_ID_TIME = "%Y%m%d_%H%M%S"
_RUN_ID_HEX = 6  # random hex chars, so two runs in one second get distinct ids
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
    meta.json. Mapping keys are written as str(), sets as sorted lists, NaN and
    infinity as null, and any other value JSON cannot hold as its str(). Every
    file is serialized before the directory is made, so a failure leaves none.
    run_id must be a single directory name, else ValueError; it defaults to the
    UTC time, a slug of the experiment name and random hex. An existing run
    directory raises FileExistsError, so no run is overwritten. The rows hold
    customer text, and results/ is gitignored.
    """
    if run_id is not None and (Path(run_id).name != run_id or run_id in {".", ".."}):
        raise ValueError(settings.texts.bad_run_id.format(run_id=run_id))
    texts = {
        _ROWS_FILE: "".join(_to_json(vars(row)) for row in report.rows),
        _SUMMARY_FILE: _to_json(summary, _INDENT),
        _META_FILE: _to_json(meta, _INDENT),
    }
    run_dir = results_dir / (run_id or _default_run_id(report.experiment_name))
    run_dir.mkdir(parents=True)
    for name, text in texts.items():
        (run_dir / name).write_text(text, encoding="utf-8")
    return run_dir


def _default_run_id(experiment_name: str) -> str:
    """Return the UTC time, a slug of the experiment name and random hex."""
    slug = _NOT_SLUG.sub("-", experiment_name.lower()).strip("-")
    stamp = datetime.now(UTC).strftime(_RUN_ID_TIME)
    return f"{stamp}_{slug}_{uuid.uuid4().hex[:_RUN_ID_HEX]}"


def _to_json(value: Mapping[str, Any], indent: int | None = None) -> str:
    """Return value as UTF-8 JSON ending with a newline, on one line unless indented."""
    text = json.dumps(
        _json_safe(value),
        indent=indent,
        default=str,
        ensure_ascii=False,
        allow_nan=False,
    )
    return text + "\n"


def _json_safe(value: object) -> object:
    """Return value with str mapping keys, sets as lists and non-finite floats None."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return sorted((_json_safe(item) for item in value), key=str)
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = ["write_run"]
