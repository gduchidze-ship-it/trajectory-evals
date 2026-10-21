"""Tests for the experiment runner's local and upload paths and its value types."""

import socket
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, NoReturn, TypedDict, cast

import pytest
from langsmith import Client, schemas
from langsmith.utils import get_env_var

from trajectory_evals.core import runner
from trajectory_evals.core.runner import (
    ExperimentConfig,
    ExperimentReport,
    ExperimentRunner,
    Target,
    build_local_client,
)
from trajectory_evals.core.settings import settings as eval_settings
from trajectory_evals.core.types import DatasetExample

_DATASET = "jailbreak-detection/sample"
_EXAMPLES = [
    DatasetExample(
        case_id=f"case-{number}",
        inputs={"text": f"message {number}"},
        outputs={"expected": number % 2 == 0},
        metadata={"channel": "sms"},
    )
    for number in range(4)
]
_METADATA = {"dataset_file": "sample.jsonl", "row_count": len(_EXAMPLES)}
_FAKE_EXPERIMENT = "upload-test-1234"
_FAKE_URL = "https://smith.example/o/1/datasets/2/compare?selectedSessions=3"


class _MessageInputs(TypedDict):
    """The inputs every sample case carries."""

    text: str


class _FlagOutputs(TypedDict):
    """What the sample target returns for one message."""

    flagged: bool


class _EvenTarget:
    """A callable-class target whose async __call__ returns a TypedDict."""

    async def __call__(self, inputs: _MessageInputs) -> _FlagOutputs:
        """Flag the message when its number is even."""
        return {"flagged": int(inputs["text"].split()[-1]) % 2 == 0}


_TARGET: Target = _EvenTarget()


class _ListSource:
    """A case source over the fixed sample examples."""

    name = "sample"

    def examples(self) -> Iterator[DatasetExample]:
        """Yield every sample example in order."""
        yield from _EXAMPLES


def _matches(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> bool:
    """Return whether the target's flag matches the label."""
    return bool(outputs["flagged"] == reference_outputs["expected"])


def _accuracy(
    outputs: list[dict[str, Any]], reference_outputs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return the share of rows whose flag matches the label."""
    hits = sum(
        output["flagged"] == reference["expected"]
        for output, reference in zip(outputs, reference_outputs, strict=True)
    )
    return {"key": "accuracy", "score": hits / len(outputs)}


def _undefined_precision(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a precision with no defined value."""
    return {"key": "precision", "score": None}


class _RecordingSink:
    """A dataset sink that records each sync and the examples it received."""

    def __init__(self, events: list[str]) -> None:
        """Share the event log with the other fakes."""
        self.events = events
        self.synced: list[tuple[str, list[DatasetExample]]] = []

    def sync(self, dataset_name: str, examples: Iterable[DatasetExample]) -> str:
        """Record the sync and return the dataset name."""
        self.events.append("sync")
        self.synced.append((dataset_name, list(examples)))
        return dataset_name

    def example_id(self, dataset_name: str, case_id: str) -> uuid.UUID:
        """Return a stable id for the case in the named dataset."""
        return uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_name}:{case_id}")


class _RecordingClient:
    """A LangSmith client stand-in that records list_examples calls."""

    def __init__(self, events: list[str]) -> None:
        """Share the event log with the other fakes."""
        self.events = events
        self.listed: list[dict[str, Any]] = []

    def list_examples(self, **kwargs: Any) -> Iterator[schemas.Example]:
        """Record the call and return the requested examples in reverse order."""
        self.events.append("list_examples")
        self.listed.append(kwargs)
        return iter(
            [
                schemas.Example(id=example_id, inputs={}, outputs={})
                for example_id in reversed(kwargs["example_ids"])
            ]
        )


def _fake_row(case_id: str, flagged: bool, expected: bool) -> dict[str, Any]:
    """Return one aevaluate result row whose run took no time and was not scored."""
    now = datetime.now(UTC)
    return {
        "run": SimpleNamespace(
            outputs={"flagged": flagged}, error=None, start_time=now, end_time=now
        ),
        "example": schemas.Example(
            id=uuid.uuid4(),
            inputs={"text": case_id},
            outputs={"expected": expected},
            metadata={"case_id": case_id},
        ),
        "evaluation_results": {"results": []},
    }


class _FakeResults:
    """The parts of AsyncExperimentResults that the runner reads."""

    experiment_name = _FAKE_EXPERIMENT

    @property
    def experiment_id(self) -> NoReturn:
        """Fail, because the runner must not read it; a local run would raise."""
        raise AssertionError("the runner read experiment_id")

    @property
    def url(self) -> NoReturn:
        """Fail, because the runner must not read it; a local run would raise."""
        raise AssertionError("the runner read url")

    async def get_comparison_url(self) -> str:
        """Return the experiment's comparison URL."""
        return _FAKE_URL

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """Yield two rows, one flagged correctly, so accuracy is 0.5."""
        yield _fake_row("case-0", flagged=True, expected=True)
        yield _fake_row("case-1", flagged=True, expected=False)


class _RecordingAevaluate:
    """An aevaluate stand-in that records its arguments."""

    def __init__(self, events: list[str]) -> None:
        """Share the event log with the other fakes."""
        self.events = events
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, target: Any, /, **kwargs: Any) -> _FakeResults:
        """Record the call and return fixed results."""
        self.events.append("aevaluate")
        self.kwargs = kwargs
        return _FakeResults()


def _clear_langsmith_env_cache() -> None:
    """Drop langsmith's cached env reads; mypy sees only get_env_var's overloads."""
    cast(Any, get_env_var).cache_clear()


@pytest.fixture
def fake_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Set a fake key with tracing on, clearing langsmith's env cache around it."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-fake-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    _clear_langsmith_env_cache()
    yield
    _clear_langsmith_env_cache()


@pytest.mark.asyncio
@pytest.mark.usefixtures("fake_langsmith_env")
@pytest.mark.parametrize(("limit", "runs"), [(None, 4), (2, 2)])
async def test_local_run_scores_a_callable_class_target_offline(
    monkeypatch: pytest.MonkeyPatch, limit: int | None, runs: int
) -> None:
    """A local run on the injected offline client scores a target and syncs nothing."""
    attempts: list[tuple[Any, ...]] = []

    def refuse(*args: Any, **kwargs: Any) -> NoReturn:
        """Record a network attempt and refuse it."""
        attempts.append(args)
        raise OSError(f"network disabled in test: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    events: list[str] = []
    experiment_runner = ExperimentRunner(
        client=build_local_client(), sink=_RecordingSink(events)
    )

    report = await experiment_runner.run(
        source=_ListSource(),
        dataset_name=_DATASET,
        target=_TARGET,
        evaluators=[_matches],
        summary_evaluators=[_accuracy, _undefined_precision],
        config=ExperimentConfig(
            experiment_prefix="local-test",
            max_concurrency=8,
            upload_results=False,
            limit=limit,
        ),
        metadata=_METADATA,
    )

    assert report == ExperimentReport(
        experiment_name="local-test (local)",
        dataset_name=_DATASET,
        url=None,
        summary={"accuracy": 1.0, "precision": None},
        runs=runs,
    )
    assert (events, attempts) == ([], [])


@pytest.mark.asyncio
async def test_upload_run_syncs_everything_then_evaluates_the_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uploaded run mirrors every case, then evaluates the dataset by name."""
    events: list[str] = []
    aevaluate = _RecordingAevaluate(events)
    monkeypatch.setattr(runner, "aevaluate", aevaluate)
    sink, client = _RecordingSink(events), _RecordingClient(events)
    evaluators, summary_evaluators = [_matches], [_accuracy]

    report = await ExperimentRunner(client=cast(Client, client), sink=sink).run(
        source=_ListSource(),
        dataset_name=_DATASET,
        target=_TARGET,
        evaluators=evaluators,
        summary_evaluators=summary_evaluators,
        config=ExperimentConfig(
            experiment_prefix="upload-test",
            max_concurrency=4,
            num_repetitions=2,
        ),
        metadata=_METADATA,
    )

    assert events == ["sync", "aevaluate"]
    assert sink.synced == [(_DATASET, _EXAMPLES)]
    assert aevaluate.kwargs == {
        "data": _DATASET,
        "evaluators": evaluators,
        "summary_evaluators": summary_evaluators,
        "metadata": _METADATA,
        "experiment_prefix": "upload-test",
        "max_concurrency": 4,
        "num_repetitions": 2,
        "client": client,
        "upload_results": True,
    }
    assert report == ExperimentReport(
        experiment_name=_FAKE_EXPERIMENT,
        dataset_name=_DATASET,
        url=_FAKE_URL,
        summary={"accuracy": 0.5},
        runs=2,
    )


@pytest.mark.asyncio
async def test_upload_run_with_limit_evaluates_the_first_cases_in_source_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A limited run syncs every case, then fetches the first ones by id in order."""
    events: list[str] = []
    aevaluate = _RecordingAevaluate(events)
    monkeypatch.setattr(runner, "aevaluate", aevaluate)
    langsmith = eval_settings.langsmith.model_copy(update={"ids_per_request": 2})
    batched = eval_settings.model_copy(update={"langsmith": langsmith})
    monkeypatch.setattr(runner, "settings", batched)
    sink, client = _RecordingSink(events), _RecordingClient(events)

    await ExperimentRunner(client=cast(Client, client), sink=sink).run(
        source=_ListSource(),
        dataset_name=_DATASET,
        target=_TARGET,
        evaluators=[_matches],
        summary_evaluators=[],
        config=ExperimentConfig(
            experiment_prefix="upload-test", max_concurrency=8, limit=3
        ),
        metadata=_METADATA,
    )

    expected_ids = [sink.example_id(_DATASET, f"case-{number}") for number in range(3)]
    assert events == ["sync", "list_examples", "list_examples", "aevaluate"]
    assert sink.synced == [(_DATASET, _EXAMPLES)]
    assert client.listed == [
        {"dataset_name": _DATASET, "example_ids": tuple(expected_ids[:2])},
        {"dataset_name": _DATASET, "example_ids": tuple(expected_ids[2:])},
    ]
    assert [example.id for example in aevaluate.kwargs["data"]] == expected_ids


@pytest.mark.parametrize("name", ["max_concurrency", "num_repetitions", "limit"])
def test_config_rejects_counts_below_one(name: str) -> None:
    """A concurrency, repetition count or limit of zero is refused at construction."""
    overrides: dict[str, Any] = {"max_concurrency": 8, name: 0}
    with pytest.raises(ValueError, match=f"{name} must be at least 1"):
        ExperimentConfig(experiment_prefix="invalid", **overrides)


def test_report_renders_a_local_run_with_an_undefined_score() -> None:
    """A local report prints its dataset, no URL and n/a for an undefined score."""
    report = ExperimentReport(
        experiment_name="detection (local)",
        dataset_name=_DATASET,
        url=None,
        summary={"accuracy": 0.5, "precision": None},
        runs=3,
    )

    assert report.render().splitlines() == [
        "experiment: detection (local)",
        "dataset: jailbreak-detection/sample",
        "url: not uploaded",
        "runs: 3",
        "accuracy: 0.5000",
        "precision: n/a",
    ]


def test_report_labels_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each report line uses the label and line template from [langsmith]."""
    labels = eval_settings.langsmith.report_labels.model_copy(update={"url": "link"})
    langsmith = eval_settings.langsmith.model_copy(
        update={"report_labels": labels, "report_line": "{label} = {value}"}
    )
    monkeypatch.setattr(
        runner,
        "settings",
        eval_settings.model_copy(update={"langsmith": langsmith}),
    )
    report = ExperimentReport(
        experiment_name="detection (local)",
        dataset_name=_DATASET,
        url=None,
        summary={"accuracy": 0.5},
        runs=3,
    )

    assert report.render().splitlines() == [
        "experiment = detection (local)",
        "dataset = jailbreak-detection/sample",
        "link = not uploaded",
        "runs = 3",
        "accuracy = 0.5000",
    ]


class _RaisingTarget:
    """A target that fails on every message, like an unreachable agent."""

    async def __call__(self, inputs: _MessageInputs) -> _FlagOutputs:
        """Raise instead of returning outputs."""
        raise RuntimeError("target failed")


async def _refuse_comparison_url(self: _FakeResults) -> NoReturn:
    """Fail, because a local run's results raise when asked for this URL."""
    raise AssertionError("the runner asked a local run for its comparison URL")


def _local_config(prefix: str, limit: int | None = None) -> ExperimentConfig:
    """Return the config of a local run under prefix over the first limit cases."""
    return ExperimentConfig(
        experiment_prefix=prefix, max_concurrency=2, upload_results=False, limit=limit
    )


@pytest.mark.asyncio
async def test_local_run_returns_rows_and_scores_its_summary_on_them(
    network_attempts: list[Any],
) -> None:
    """A local run keeps one row per case, and its summary is computed from them."""
    experiment_runner = ExperimentRunner(
        client=build_local_client(), sink=_RecordingSink([])
    )

    report = await experiment_runner.run(
        source=_ListSource(),
        dataset_name=_DATASET,
        target=_TARGET,
        evaluators=[_matches],
        summary_evaluators=[_accuracy, _undefined_precision],
        config=_local_config("rows-test"),
        metadata=_METADATA,
    )

    rows = sorted(report.rows, key=lambda row: row.case_id)
    outputs = [dict(row.outputs) for row in rows]
    references = [dict(row.reference_outputs) for row in rows]
    assert [row.case_id for row in rows] == [case["case_id"] for case in _EXAMPLES]
    assert outputs == [{"flagged": number % 2 == 0} for number in range(4)]
    assert references == [case["outputs"] for case in _EXAMPLES]
    assert {(row.feedback["_matches"], row.error) for row in rows} == {(True, None)}
    assert all(row.latency_ms is not None and row.latency_ms >= 0 for row in rows)
    assert report.summary == {
        "accuracy": _accuracy(outputs, references)["score"],
        "precision": None,
    }
    assert network_attempts == []


@pytest.mark.asyncio
async def test_local_row_of_a_raising_target_keeps_the_errored_outputs(
    network_attempts: list[Any],
) -> None:
    """A raised target's row holds its error and the outputs LangSmith substitutes."""
    experiment_runner = ExperimentRunner(
        client=build_local_client(), sink=_RecordingSink([])
    )

    report = await experiment_runner.run(
        source=_ListSource(),
        dataset_name=_DATASET,
        target=_RaisingTarget(),
        evaluators=[],
        summary_evaluators=[],
        config=_local_config("errored-test", limit=1),
        metadata=_METADATA,
    )

    [row] = report.rows
    assert row.outputs == {eval_settings.evaluators.errored_output_key: None}
    assert "target failed" in (row.error or "")
    assert network_attempts == []


@pytest.mark.asyncio
async def test_local_run_never_asks_for_the_experiment_id_or_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local run names itself and reads no experiment id, URL or comparison URL."""
    monkeypatch.setattr(runner, "aevaluate", _RecordingAevaluate([]))
    monkeypatch.setattr(_FakeResults, "get_comparison_url", _refuse_comparison_url)
    experiment_runner = ExperimentRunner(
        client=build_local_client(), sink=_RecordingSink([])
    )

    report = await experiment_runner.run(
        source=_ListSource(),
        dataset_name=_DATASET,
        target=_TARGET,
        evaluators=[_matches],
        summary_evaluators=[_accuracy],
        config=_local_config("local-test"),
        metadata=_METADATA,
    )

    assert (report.experiment_name, report.url) == ("local-test (local)", None)
    assert [row.case_id for row in report.rows] == ["case-0", "case-1"]
    assert report.summary == {"accuracy": 0.5}
