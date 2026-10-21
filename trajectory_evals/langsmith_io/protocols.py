"""The LangSmith client calls that pulling, pushing, harvesting and scoring make."""

from collections.abc import AsyncIterable, Iterable
from datetime import datetime
from typing import Protocol
from uuid import UUID

from langsmith.schemas import Example, TracerSession
from langsmith.types import Run, RunSelectField

from trajectory_evals.core.dataset import _LangSmithClient


class RunQuery(Protocol):
    """The v2 runs resource: an async paginated query of a project's runs."""

    def query(
        self,
        *,
        project_ids: list[str],
        is_root: bool,
        min_start_time: datetime,
        selects: list[RunSelectField],
        page_size: int,
        filter: str = ...,
        has_error: bool = ...,
    ) -> AsyncIterable[Run]:
        """Yield the matching runs page by page; only selected fields are set."""


class LangSmithClient(_LangSmithClient, Protocol):
    """A langsmith.Client, as the functions of this package call it.

    Real use passes langsmith.Client(), never one built with
    tracing_sampling_rate below 1, which silently drops that share of feedback.
    """

    @property
    def runs(self) -> RunQuery:
        """Return the v2 runs resource, built from this client's endpoint and key."""

    @property
    def tracing_sample_rate(self) -> float | None:
        """Return the share of traces and feedback sent, None meaning all."""

    def list_examples(
        self, *, dataset_id: UUID, as_of: datetime | str | None = None
    ) -> Iterable[Example]:
        """Yield the dataset's examples, as of a version time or tag when given."""

    async def aread_project(self, *, project_name: str) -> TracerSession:
        """Return the named project, raising LangSmithNotFoundError when missing."""

    def create_feedback(
        self,
        run_id: str | None,
        key: str,
        *,
        score: float | bool,
        comment: str | None,
        session_id: str | None,
        trace_id: str | None,
        feedback_id: UUID,
        start_time: datetime | None,
        feedback_source_type: str,
        extend_trace_retention: bool,
    ) -> object:
        """Record feedback on a run; without extended retention it posts and waits."""

    def flush(self) -> None:
        """Block until queued traces and feedback are sent."""


__all__ = ["LangSmithClient", "RunQuery"]
