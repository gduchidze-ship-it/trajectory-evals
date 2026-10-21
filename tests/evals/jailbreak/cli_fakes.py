"""Labelled JSONL files that the command-line tests write and run on."""

import json
from pathlib import Path

from evals.jailbreak.types import ExpectedAction


def case_row(
    case_id: str, text: str, expected_action: ExpectedAction, channel: str = "sms"
) -> dict[str, str]:
    """Return one labelled JSONL row in the prod export's shape."""
    return {
        "id": case_id,
        "input_text": text,
        "expected_action": expected_action,
        "channel": channel,
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write rows to path as JSON lines and return the path."""
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    return path
