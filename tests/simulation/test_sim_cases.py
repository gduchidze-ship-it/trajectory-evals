"""Tests for scenario rows read from JSONL and red-team cases built from personas."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from trajectory_evals.core.jsonl import CaseFormatError
from trajectory_evals.simulation import Persona, ScenarioSource, redteam_cases
from trajectory_evals.simulation.settings import settings

PERSONAS = {
    "calm": Persona("calm", "benign", "Ask politely."),
    "crescendo": Persona("crescendo", "attack", "Escalate."),
    "roleplay": Persona("roleplay", "attack", "Pretend."),
}


def _write(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    """Write rows to scenarios.jsonl, one JSON object per line, and return its path."""
    path = tmp_path / "scenarios.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_rows_become_examples_and_other_keys_metadata(tmp_path: Path) -> None:
    """Named keys fill inputs and outputs; absent ones default; the rest is metadata."""
    full = {
        "id": "late-order",
        "persona": "benign_frustrated",
        "scenario": "Your order is late.",
        "opening_message": " Where is my bike? ",
        "max_turns": 3,
        "context": {"channel": "sms"},
        "expected": {"not_contains": ["refund"], "action": "ESCALATE"},
        "risk": "high",
    }
    minimal = {"id": "hours", "persona": "benign_curious", "scenario": "Ask hours."}
    source = ScenarioSource(_write(tmp_path, full, minimal))

    examples = list(source.examples())

    assert source.name == "scenarios"
    assert examples == [
        {
            "case_id": "late-order",
            "inputs": {
                "persona": "benign_frustrated",
                "scenario": "Your order is late.",
                "opening_message": " Where is my bike? ",
                "max_turns": 3,
                "context": {"channel": "sms"},
            },
            "outputs": {"not_contains": ["refund"], "action": "ESCALATE"},
            "metadata": {"risk": "high"},
        },
        {
            "case_id": "hours",
            "inputs": {
                "persona": "benign_curious",
                "scenario": "Ask hours.",
                "opening_message": None,
                "max_turns": None,
                "context": {},
            },
            "outputs": {},
            "metadata": {},
        },
    ]


ROW = {"id": "a", "persona": "crescendo", "scenario": "Get a discount."}
BAD_ROWS: dict[str, tuple[dict[str, Any], str]] = {
    "no_persona": ({"id": "b", "scenario": "x"}, "persona: Field required"),
    "zero_turns": ({**ROW, "id": "b", "max_turns": 0}, "max_turns: Input should be"),
    "context_list": ({**ROW, "id": "b", "context": []}, "context: Input should be"),
    "bad_expected": (
        {**ROW, "id": "b", "expected": {"contains": "x"}},
        "expected.contains: Input should be a valid list",
    ),
    "repeated_id": (ROW, "case_id 'a' appears more than once"),
}


@pytest.mark.parametrize(("row", "reason"), BAD_ROWS.values(), ids=BAD_ROWS)
def test_a_bad_row_names_its_path_line_and_reason(
    tmp_path: Path, row: dict[str, Any], reason: str
) -> None:
    """Every kind of bad row raises CaseFormatError as path:line: reason."""
    path = _write(tmp_path, ROW, row)
    with pytest.raises(CaseFormatError, match=f"^{re.escape(str(path))}:2: {reason}"):
        list(ScenarioSource(path).examples())


def test_redteam_cases_cross_every_attack_persona_with_each_attempt() -> None:
    """names=None takes every attack persona; ids are persona:attempt, stable."""
    source = redteam_cases(PERSONAS, names=None, attempts=2)
    examples = list(source.examples())

    assert source.name == "redteam"
    assert [example["case_id"] for example in examples] == [
        "crescendo:1",
        "crescendo:2",
        "roleplay:1",
        "roleplay:2",
    ]
    assert examples[0]["inputs"] == {
        "persona": "crescendo",
        "scenario": settings.prompts.redteam_scenario,
        "opening_message": None,
        "max_turns": None,
        "context": {},
    }
    assert list(redteam_cases(PERSONAS, names=None, attempts=2).examples()) == examples


def test_redteam_cases_keep_the_named_order_once_each() -> None:
    """Named personas run in the order given, a repeated name only once."""
    source = redteam_cases(
        PERSONAS, names=["roleplay", "crescendo", "roleplay"], attempts=1
    )
    ids = [example["case_id"] for example in source.examples()]
    assert ids == ["roleplay:1", "crescendo:1"]


@pytest.mark.parametrize(
    ("names", "attempts", "message"),
    [
        (["calm", "nope"], 1, r"must be attack personas, but \['calm', 'nope'\]"),
        (None, 0, "attempts must be at least 1, got 0"),
    ],
    ids=["not_attackers", "no_attempts"],
)
def test_redteam_cases_reject_what_cannot_attack(
    names: list[str] | None, attempts: int, message: str
) -> None:
    """A benign or unknown name, or fewer than one attempt, fails up front."""
    with pytest.raises(ValueError, match=message):
        redteam_cases(PERSONAS, names=names, attempts=attempts)
