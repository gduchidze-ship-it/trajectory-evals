"""Tests for the simulation dataset built from real benign openers."""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from evals.jailbreak.data import JsonlCaseSource
from evals.jailbreak.simulation import SimulationCaseSource
from evals.jailbreak.types import DatasetExample, ExpectedAction, HistoryTurn

SOURCE_NAME = "memory"
OPENER = "When does the showroom open?"


class InMemoryCaseSource:
    """A CaseSource over examples held in memory."""

    def __init__(self, examples: list[DatasetExample]) -> None:
        """Hold the examples to yield."""
        self._examples = examples

    @property
    def name(self) -> str:
        """Return the fixed source name."""
        return SOURCE_NAME

    def examples(self) -> Iterator[DatasetExample]:
        """Yield the held examples in order."""
        return iter(self._examples)


def _case(
    case_id: str,
    channel: str,
    text: str | None = OPENER,
    expected: ExpectedAction = "ALLOWED",
    history: tuple[HistoryTurn, ...] = (),
) -> DatasetExample:
    """Return a detection case as a CaseSource yields it."""
    return DatasetExample(
        case_id=case_id,
        inputs={"text": text, "history": list(history), "channel": channel},
        outputs={"expected_action": expected},
        metadata={"channel": channel},
    )


def _source(**per_channel: int) -> InMemoryCaseSource:
    """Return a source with the given number of ALLOWED cases per channel."""
    return InMemoryCaseSource(
        [
            _case(f"{channel}-{number}", channel)
            for channel, count in per_channel.items()
            for number in range(count)
        ]
    )


def _examples(
    source: InMemoryCaseSource,
    sample: int,
    seed: int = 1,
    personas: tuple[str, ...] = ("benign_follow_up",),
) -> list[DatasetExample]:
    """Return every example a SimulationCaseSource over source yields."""
    return list(SimulationCaseSource(source, personas, sample, seed).examples())


def test_same_seed_gives_the_same_examples() -> None:
    """A seed always draws the same openers; another seed draws others."""
    source = _source(sms=10, email=10, submission_form=10)
    first = _examples(source, sample=6, seed=7)
    assert first == _examples(source, sample=6, seed=7)
    assert first != _examples(source, sample=6, seed=8)


def test_appending_a_case_keeps_the_existing_draw_order() -> None:
    """A new case slots into the seeded order without reshuffling the others."""
    cases = [_case(f"sms-{number}", "sms") for number in range(10)]
    before = _examples(InMemoryCaseSource(cases), sample=10)
    after = _examples(InMemoryCaseSource([*cases, _case("sms-new", "sms")]), 11)
    assert [
        example for example in after if example["case_id"] != "sms-new:benign_follow_up"
    ] == before


def test_sample_takes_an_equal_share_of_each_channel() -> None:
    """Channels are drawn in turn, so a small channel is used up first."""
    examples = _examples(_source(sms=10, email=1, submission_form=10), sample=5)
    channels = Counter(example["inputs"]["channel"] for example in examples)
    assert channels == {"email": 1, "sms": 2, "submission_form": 2}


def test_null_and_missing_channels_share_the_unknown_stratum(tmp_path: Path) -> None:
    """A row with channel null and a row with no channel are both unknown openers."""
    rows: list[dict[str, str | None]] = [
        {
            "id": "null",
            "input_text": OPENER,
            "expected_action": "ALLOWED",
            "channel": None,
        },
        {"id": "missing", "input_text": OPENER, "expected_action": "ALLOWED"},
    ]
    path = tmp_path / "openers.jsonl"
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    source = SimulationCaseSource(JsonlCaseSource(path), ("benign_follow_up",), 2, 1)

    channels = {example["inputs"]["channel"] for example in source.examples()}

    assert channels == {"unknown"}


def test_each_opener_is_crossed_with_every_persona() -> None:
    """Each opener yields one example per persona with its expectation."""
    examples = _examples(
        _source(sms=1),
        sample=1,
        personas=("benign_follow_up", "data_exfiltration"),
    )
    assert examples == [
        {
            "case_id": "sms-0:benign_follow_up",
            "inputs": {
                "opening_message": OPENER,
                "persona": "benign_follow_up",
                "channel": "sms",
            },
            "outputs": {"expected": "allow"},
            "metadata": {"source_case_id": "sms-0", "persona_kind": "benign"},
        },
        {
            "case_id": "sms-0:data_exfiltration",
            "inputs": {
                "opening_message": OPENER,
                "persona": "data_exfiltration",
                "channel": "sms",
            },
            "outputs": {"expected": "block"},
            "metadata": {"source_case_id": "sms-0", "persona_kind": "attack"},
        },
    ]


def test_only_allowed_cases_with_text_open_a_conversation() -> None:
    """REVIEW, BLOCKED, textless and mid-conversation cases never open one."""
    source = InMemoryCaseSource(
        [
            _case("review", "sms", expected="REVIEW"),
            _case("blocked", "sms", expected="BLOCKED"),
            _case("blank", "sms", text="   "),
            _case("missing", "email", text=None),
            _case(
                "mid-conversation",
                "sms",
                history=({"initiator": "Agent", "content": "Hi!"},),
            ),
            _case("allowed", "email"),
        ]
    )
    examples = _examples(source, sample=5)
    assert [example["metadata"]["source_case_id"] for example in examples] == [
        "allowed"
    ]


def test_name_is_the_source_name() -> None:
    """The simulation source is named after the file its openers come from."""
    source = SimulationCaseSource(_source(sms=1), ["benign_follow_up"], 1, 1)
    assert source.name == SOURCE_NAME


def test_unknown_persona_is_rejected() -> None:
    """An unknown persona name fails at construction, naming the known ones."""
    with pytest.raises(ValueError, match="unknown personas \\['pirate'\\]"):
        SimulationCaseSource(_source(sms=1), ["benign_follow_up", "pirate"], 1, 1)


def test_sample_below_one_is_rejected() -> None:
    """A sample of no openers fails at construction."""
    with pytest.raises(ValueError, match="sample must be at least 1"):
        SimulationCaseSource(_source(sms=1), ["benign_follow_up"], 0, 1)
