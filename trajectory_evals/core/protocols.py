"""Structural interfaces that the harness takes as collaborators."""

from collections.abc import Iterable, Iterator
from typing import Protocol
from uuid import UUID

from trajectory_evals.core.types import DatasetExample


class CaseSource(Protocol):
    """A named file of labelled cases."""

    @property
    def name(self) -> str:
        """Return the stem of the file the cases come from."""

    def examples(self) -> Iterator[DatasetExample]:
        """Yield every case as a dataset example."""


class DatasetSink(Protocol):
    """A store that mirrors examples into a named dataset."""

    def sync(self, dataset_name: str, examples: Iterable[DatasetExample]) -> str:
        """Mirror examples into the named dataset and return its name."""

    def example_id(self, dataset_name: str, case_id: str) -> UUID:
        """Return the id the named dataset stores the case under."""


__all__ = ["CaseSource", "DatasetSink"]
