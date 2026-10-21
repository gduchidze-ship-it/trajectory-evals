"""Structural interfaces that the jailbreak evaluations take as collaborators."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol
from uuid import UUID

from trajectory_evals.suites.jailbreak.types import DatasetExample
from src.toolsets.jailbreak_detection.types import (
    ConversationTurn,
    JailbreakDetectionResult,
)


class Detector(Protocol):
    """Screens one inbound message the way detect_jailbreak does."""

    async def check(
        self, text: str | None, history: Sequence[ConversationTurn] | None = None
    ) -> JailbreakDetectionResult:
        """Return the verdict on text, or on the newest customer turn of history."""


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
