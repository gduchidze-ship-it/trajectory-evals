"""Idempotent mirroring of labelled examples into LangSmith datasets."""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Any, Protocol
from uuid import NAMESPACE_DNS, UUID, uuid5

from langsmith.schemas import Dataset, Example
from langsmith.utils import LangSmithNotFoundError

from trajectory_evals.core.settings import settings
from trajectory_evals.core.types import DatasetExample


class _LangSmithClient(Protocol):
    """The LangSmith client calls that mirroring a dataset makes."""

    def read_dataset(self, *, dataset_name: str) -> Dataset:
        """Return the named dataset, raising LangSmithNotFoundError when missing."""

    def create_dataset(self, dataset_name: str) -> Dataset:
        """Create and return an empty dataset."""

    def list_examples(self, *, dataset_id: UUID) -> Iterable[Example]:
        """Yield the dataset's examples that are not deleted."""

    def create_examples(
        self, *, dataset_id: UUID, examples: Sequence[dict[str, Any]]
    ) -> object:
        """Create the given example bodies, splitting them by size itself."""

    def update_examples(
        self, *, dataset_id: UUID, updates: Sequence[dict[str, Any]]
    ) -> object:
        """Overwrite the examples the given bodies name."""

    def delete_examples(self, example_ids: Sequence[UUID]) -> None:
        """Soft-delete the examples with these IDs."""


@dataclass(frozen=True)
class SyncPlan:
    """The example writes that bring a dataset in line with its source."""

    creates: tuple[dict[str, Any], ...]
    updates: tuple[dict[str, Any], ...]
    deletes: tuple[UUID, ...]


def build_payloads(
    dataset_name: str, examples: Iterable[DatasetExample], *, namespace: str
) -> dict[UUID, dict[str, Any]]:
    """Return every example's LangSmith body by stable ID, rejecting repeated IDs."""
    payloads: dict[UUID, dict[str, Any]] = {}
    for example in examples:
        payload = _example_payload(namespace, dataset_name, example)
        if payload["id"] in payloads:
            message = settings.texts.duplicate_case_id
            raise ValueError(message.format(case_id=example["case_id"]))
        payloads[payload["id"]] = payload
    return payloads


def plan_sync(
    payloads: Mapping[UUID, dict[str, Any]], stored_hashes: Mapping[UUID, str | None]
) -> SyncPlan:
    """Diff payloads against the hashes a dataset stores and list the writes needed."""
    hash_key = settings.data.content_hash_key
    creates = tuple(
        payload
        for example_id, payload in payloads.items()
        if example_id not in stored_hashes
    )
    updates = tuple(
        payload
        for example_id, payload in payloads.items()
        if example_id in stored_hashes
        and stored_hashes[example_id] != payload["metadata"][hash_key]
    )
    deletes = tuple(
        stored_id for stored_id in stored_hashes if stored_id not in payloads
    )
    return SyncPlan(creates=creates, updates=updates, deletes=deletes)


class LangSmithDatasetSync:
    """Mirrors examples into a LangSmith dataset, writing only what changed.

    An example's ID is uuid5 of "<dataset>:<case_id>" under uuid5(NAMESPACE_DNS,
    namespace), so a different namespace gives every example a new ID.
    """

    def __init__(self, client: _LangSmithClient, *, namespace: str) -> None:
        """Mirror datasets through client under namespace; nothing is called yet."""
        self._client = client
        self._namespace = namespace

    def sync(self, dataset_name: str, examples: Iterable[DatasetExample]) -> str:
        """Mirror examples into the named dataset and return its name."""
        payloads = build_payloads(dataset_name, examples, namespace=self._namespace)
        dataset_id = self._dataset_id(dataset_name)
        self._apply(dataset_id, plan_sync(payloads, self._stored_hashes(dataset_id)))
        return dataset_name

    def example_id(self, dataset_name: str, case_id: str) -> UUID:
        """Return the id the named dataset stores the case under."""
        return _example_id(self._namespace, dataset_name, case_id)

    def _dataset_id(self, dataset_name: str) -> UUID:
        """Return the ID of the named dataset, creating the dataset when missing."""
        try:
            dataset = self._client.read_dataset(dataset_name=dataset_name)
        except LangSmithNotFoundError:
            dataset = self._client.create_dataset(dataset_name)
        return dataset.id

    def _stored_hashes(self, dataset_id: UUID) -> dict[UUID, str | None]:
        """Return the content hash stored on each example already in the dataset."""
        return {
            example.id: (example.metadata or {}).get(settings.data.content_hash_key)
            for example in self._client.list_examples(dataset_id=dataset_id)
        }

    def _apply(self, dataset_id: UUID, plan: SyncPlan) -> None:
        """Send the creates at once and the updates and deletes in batches."""
        if plan.creates:
            self._client.create_examples(dataset_id=dataset_id, examples=plan.creates)
        for updates in batched(plan.updates, settings.data.update_batch_size):
            self._client.update_examples(dataset_id=dataset_id, updates=updates)
        for deletes in batched(plan.deletes, settings.data.delete_batch_size):
            self._client.delete_examples(example_ids=deletes)


def _example_id(namespace: str, dataset_name: str, case_id: str) -> UUID:
    """Return the stable ID of a case in the named dataset under namespace."""
    return uuid5(uuid5(NAMESPACE_DNS, namespace), f"{dataset_name}:{case_id}")


def _example_payload(
    namespace: str, dataset_name: str, example: DatasetExample
) -> dict[str, Any]:
    """Return the LangSmith example body, with its stable ID, case_id and hash."""
    case_id = example["case_id"]
    return {
        "id": _example_id(namespace, dataset_name, case_id),
        "inputs": example["inputs"],
        "outputs": example["outputs"],
        "metadata": {
            **example["metadata"],
            settings.data.case_id_key: case_id,
            settings.data.content_hash_key: _content_hash(example),
        },
    }


def _content_hash(example: DatasetExample) -> str:
    """Return the sha256 of the inputs, outputs and metadata as canonical JSON."""
    content = [example["inputs"], example["outputs"], example["metadata"]]
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "LangSmithDatasetSync",
    "SyncPlan",
    "_LangSmithClient",
    "build_payloads",
    "plan_sync",
]
