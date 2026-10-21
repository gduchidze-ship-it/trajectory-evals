"""JSONL files of LangSmith data, written only where git cannot pick them up."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from trajectory_evals.core.settings import REPO_ROOT
from trajectory_evals.langsmith_io.settings import settings

# .gitignore anchors /data/, so nothing under it is ever committed.
DATA_DIR = REPO_ROOT / "data"


def check_out(out: Path, *, allow_tracked: bool) -> None:
    """Refuse an existing file, and one outside DATA_DIR unless allow_tracked.

    Dev tracing projects and datasets hold unmasked customer data, so the
    default is the one folder .gitignore is known to cover. This checks the
    resolved path, not git itself, so a symlink under DATA_DIR is judged by
    where it leads, and any path outside the repo needs allow_tracked.
    """
    texts = settings.texts
    if out.exists():
        raise FileExistsError(texts.exists.format(path=out))
    if not allow_tracked and not out.resolve().is_relative_to(DATA_DIR):
        raise ValueError(texts.not_ignored.format(path=out, data_dir=DATA_DIR))


def write_rows(out: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write rows to a new UTF-8 JSONL file at out, creating its folder."""
    out.parent.mkdir(parents=True, exist_ok=True)
    # "x" refuses a file that appeared since check_out, rather than replacing it.
    with out.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


__all__ = ["DATA_DIR", "check_out", "write_rows"]
