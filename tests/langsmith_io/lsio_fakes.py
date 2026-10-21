"""An in-memory LangSmith client, canned root runs and golden JSONL rows."""

import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from langsmith.schemas import Example, TracerSession
from langsmith.types import Run
from langsmith.types.run import FeedbackStats

from tests.core.test_core_dataset import FakeLangSmithClient

PROJECT = "dev-sales"
PROJECT_ID = "6f1c0d4e-1a2b-4c3d-8e9f-0a1b2c3d4e5f"
SINCE = datetime(2026, 10, 1, tzinfo=UTC)
GOLDEN_ROWS: list[dict[str, Any]] = [
    {
        "id": "price-match",
        "input": "Do you price match?",
        "expected": {"contains": ["price match"]},
        "risk": "low",
    },
    {
        "id": "refund",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Refund order A1."},
        ],
        "context": {"channel": "sms"},
        "expected": {"tools": ["refund_order"], "criteria": ["Confirms it."]},
        "risk": "high",
        "split": "holdout",
    },
]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write rows as a JSONL file at path and return the path."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return the rows of the JSONL file at path."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FakeRuns:
    """The v2 runs resource over canned runs, recording each query and each yield."""

    def __init__(self, runs: Sequence[Run]) -> None:
        """Serve runs in order to every query."""
        self._runs = list(runs)
        self.queries: list[dict[str, Any]] = []
        self.yielded = 0

    def query(self, **arguments: Any) -> AsyncIterator[Run]:
        """Record the query's arguments and return its async paginator."""
        self.queries.append(arguments)
        return self._paginate()

    async def _paginate(self) -> AsyncIterator[Run]:
        """Yield every canned run, counting each."""
        for run in self._runs:
            self.yielded += 1
            yield run


class FakeClient(FakeLangSmithClient):
    """Core's in-memory datasets plus canned runs, recorded feedback and flushes."""

    def __init__(
        self, runs: Sequence[Run] = (), *, tracing_sample_rate: float | None = None
    ) -> None:
        """Serve runs from one project; nothing is recorded yet."""
        super().__init__()
        self.runs = FakeRuns(runs)
        self.tracing_sample_rate = tracing_sample_rate
        self.projects: list[str] = []
        self.as_of: list[datetime | str | None] = []
        self.feedback: list[dict[str, Any]] = []
        self.flushes = 0

    def list_examples(  # type: ignore[override]
        self, *, dataset_id: UUID, as_of: datetime | str | None = None
    ) -> Iterator[Example]:
        """Record as_of and yield the dataset's live examples."""
        self.as_of.append(as_of)
        return super().list_examples(dataset_id=dataset_id)

    async def aread_project(self, *, project_name: str) -> TracerSession:
        """Record the name and return the project every run belongs to."""
        self.projects.append(project_name)
        return TracerSession(
            id=UUID(PROJECT_ID),
            name=project_name,
            tenant_id=uuid4(),
            reference_dataset_id=None,
        )

    def create_feedback(self, run_id: str | None, key: str, **fields: Any) -> None:
        """Record the feedback's run, key and every other argument."""
        self.feedback.append({"run_id": run_id, "key": key, **fields})

    def flush(self) -> None:
        """Count the flush."""
        self.flushes += 1


def make_run(
    inputs: Mapping[str, Any] | None,
    outputs: Mapping[str, Any] | None = None,
    *,
    feedback: Mapping[str, float] | None = None,
    error: str | None = None,
) -> Run:
    """Return a root run of PROJECT_ID with a fresh id and trace id."""
    return Run(
        id=str(uuid4()),
        trace_id=str(uuid4()),
        project_id=PROJECT_ID,
        start_time=SINCE,
        inputs=dict(inputs) if inputs is not None else None,
        outputs=dict(outputs) if outputs is not None else None,
        feedback_stats={
            key: FeedbackStats(avg=avg, n=1) for key, avg in (feedback or {}).items()
        },
        error=error,
    )


__all__ = [
    "GOLDEN_ROWS",
    "PROJECT",
    "PROJECT_ID",
    "SINCE",
    "FakeClient",
    "FakeRuns",
    "make_run",
    "read_jsonl",
    "write_jsonl",
]
