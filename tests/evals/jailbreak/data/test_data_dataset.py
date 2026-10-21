"""Tests for planning and applying the LangSmith dataset mirror."""

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from langsmith.schemas import Dataset, Example, ExampleCreate, ExampleUpdate
from langsmith.utils import LangSmithConflictError, LangSmithNotFoundError

from evals.jailbreak.data import LangSmithDatasetSync
from evals.jailbreak.data.dataset import SyncPlan, build_payloads, plan_sync
from evals.jailbreak.protocols import DatasetSink
from evals.jailbreak.types import DatasetExample

DATASET = "jailbreak-detection/prod_sample"
SERVER_METADATA = {"dataset_split": ["base"]}
PINNED_ID = UUID("3f031a4d-b831-5463-94aa-a7b6595fcb69")
PINNED_HASH = "5de54a8f4a82c58d83483cf24e3e7168b4e92997dada978d364a98c01bf87cbd"


def _example(case_id: str, text: str = "Hello", **metadata: Any) -> DatasetExample:
    """A single-message ALLOWED example with the given text and metadata."""
    return DatasetExample(
        case_id=case_id,
        inputs={"text": text, "history": [], "channel": metadata.get("channel")},
        outputs={"expected_action": "ALLOWED"},
        metadata=metadata,
    )


def _id_of(case_id: str) -> UUID:
    """The ID the sink reports for a case in DATASET."""
    return LangSmithDatasetSync(FakeLangSmithClient()).example_id(DATASET, case_id)


def _stored(payloads: dict[UUID, dict[str, Any]]) -> dict[UUID, str | None]:
    """The ID-to-hash map a dataset holds once these payloads are created."""
    return {
        example_id: body["metadata"]["content_hash"]
        for example_id, body in payloads.items()
    }


class FakeLangSmithClient:
    """An in-memory LangSmith that soft-deletes and records each write's size."""

    def __init__(self) -> None:
        """Start with no datasets, no examples and no recorded writes."""
        self.datasets: dict[str, Dataset] = {}
        self.examples: dict[UUID, Example] = {}
        self.deleted: set[UUID] = set()
        self.writes: list[tuple[str, int]] = []

    def read_dataset(self, *, dataset_name: str) -> Dataset:
        """Return the named dataset, or raise LangSmith's not-found error."""
        if dataset_name not in self.datasets:
            raise LangSmithNotFoundError(f"Dataset {dataset_name} not found")
        return self.datasets[dataset_name]

    def create_dataset(self, dataset_name: str) -> Dataset:
        """Create and return an empty dataset."""
        self.writes.append(("create_dataset", 1))
        dataset = Dataset(id=uuid4(), name=dataset_name, created_at=datetime.now(UTC))
        self.datasets[dataset_name] = dataset
        return dataset

    def list_examples(self, *, dataset_id: UUID) -> Iterator[Example]:
        """Yield the dataset's examples that are not deleted."""
        live = [
            example
            for example_id, example in self.examples.items()
            if example_id not in self.deleted
        ]
        return (example for example in live if example.dataset_id == dataset_id)

    def create_examples(
        self, *, dataset_id: UUID, examples: Sequence[dict[str, Any]]
    ) -> None:
        """Store new bodies; an empty call or an ID ever used, deleted too, fails."""
        if not examples:
            raise ValueError("Must specify either 'examples' or 'inputs.'")
        bodies = [ExampleCreate(**body) for body in examples]
        if any(body.id in self.examples for body in bodies):
            raise LangSmithConflictError("Example already exists")
        self.writes.append(("create_examples", len(bodies)))
        for body in bodies:
            self._store(dataset_id, body)

    def update_examples(
        self, *, dataset_id: UUID, updates: Sequence[dict[str, Any]]
    ) -> None:
        """Overwrite each live example with its update body."""
        self.writes.append(("update_examples", len(updates)))
        for body in updates:
            update = ExampleUpdate(**body)
            assert update.id in self.examples and update.id not in self.deleted
            self._store(dataset_id, update)

    def delete_examples(self, example_ids: Sequence[UUID]) -> None:
        """Soft-delete each live example, hiding it but keeping its ID taken."""
        self.writes.append(("delete_examples", len(example_ids)))
        assert self.deleted.isdisjoint(example_ids)
        self.deleted.update(example_ids)

    def stored_texts(self, dataset_name: str) -> dict[str, Any]:
        """Return the text of each live example in the dataset, by its case_id."""
        dataset_id = self.datasets[dataset_name].id
        return {
            (example.metadata or {})["case_id"]: (example.inputs or {})["text"]
            for example in self.list_examples(dataset_id=dataset_id)
        }

    def _store(self, dataset_id: UUID, body: ExampleCreate | ExampleUpdate) -> None:
        """Keep a body as a stored example, with the server's metadata merged in."""
        assert body.id is not None
        self.examples[body.id] = Example(
            id=body.id,
            dataset_id=dataset_id,
            inputs=body.inputs,
            outputs=body.outputs,
            metadata={**(body.metadata or {}), **SERVER_METADATA},
        )


def test_plan_creates_every_example_for_an_empty_dataset() -> None:
    """Every example is created with a stable ID, its case_id and a content hash."""
    payloads = build_payloads(DATASET, [_example("a"), _example("b")])
    plan = plan_sync(payloads, {})
    assert (len(plan.creates), plan.updates, plan.deletes) == (2, (), ())
    assert [body["id"] for body in plan.creates] == [_id_of("a"), _id_of("b")]
    assert [body["metadata"]["case_id"] for body in plan.creates] == ["a", "b"]
    assert all(len(body["metadata"]["content_hash"]) == 64 for body in plan.creates)


def test_example_id_and_content_hash_are_pinned() -> None:
    """A fixed example keeps the ID and hash that LangSmith already stores."""
    sink: DatasetSink = LangSmithDatasetSync(FakeLangSmithClient())
    [(example_id, body)] = build_payloads(
        DATASET, [_example("row-1", channel="sms")]
    ).items()
    assert example_id == sink.example_id(DATASET, "row-1") == PINNED_ID
    assert body["metadata"]["content_hash"] == PINNED_HASH


def test_ids_are_scoped_to_the_dataset_name() -> None:
    """The same case gets a different ID in a different dataset."""
    simulation = build_payloads("jailbreak-simulation/prod_sample", [_example("a")])
    assert _id_of("a") not in simulation


@pytest.mark.parametrize(
    "changed",
    [_example("a", text="Changed"), _example("a", channel="email")],
    ids=["inputs", "metadata"],
)
def test_plan_updates_a_changed_example(changed: DatasetExample) -> None:
    """A stored example whose content changed is updated, not recreated."""
    stored = _stored(build_payloads(DATASET, [_example("a")]))
    plan = plan_sync(build_payloads(DATASET, [changed]), stored)
    assert (plan.creates, plan.deletes) == ((), ())
    assert [body["id"] for body in plan.updates] == [_id_of("a")]


def test_plan_deletes_an_example_no_longer_present() -> None:
    """A stored example missing from the source is deleted by its exact ID."""
    stored = _stored(build_payloads(DATASET, [_example("a"), _example("b")]))
    plan = plan_sync(build_payloads(DATASET, [_example("a")]), stored)
    assert plan == SyncPlan(creates=(), updates=(), deletes=(_id_of("b"),))


def test_plan_is_empty_when_nothing_changed() -> None:
    """Identical examples produce no writes."""
    payloads = build_payloads(DATASET, [_example("a"), _example("b", channel="sms")])
    assert plan_sync(payloads, _stored(payloads)) == SyncPlan((), (), ())


def test_build_rejects_a_repeated_case_id() -> None:
    """Two examples with one case_id would share an ID, so building refuses them."""
    with pytest.raises(ValueError, match="'a'"):
        build_payloads(DATASET, [_example("a"), _example("a", text="Other")])


def test_bad_examples_fail_before_any_client_call() -> None:
    """A repeated case_id is rejected before the dataset is even created."""
    client = FakeLangSmithClient()
    with pytest.raises(ValueError):
        LangSmithDatasetSync(client).sync(DATASET, [_example("a"), _example("a")])
    assert (client.datasets, client.writes) == ({}, [])


def test_second_identical_sync_writes_nothing() -> None:
    """Re-syncing the same examples makes no create, update or delete call."""
    client = FakeLangSmithClient()
    sync = LangSmithDatasetSync(client)
    examples = [_example("a"), _example("b")]
    assert sync.sync(DATASET, examples) == DATASET
    writes_after_first = list(client.writes)
    sync.sync(DATASET, examples)
    assert client.writes == writes_after_first


def test_sync_applies_creates_updates_and_deletes() -> None:
    """A changed source creates the new, updates the changed and deletes the gone."""
    client = FakeLangSmithClient()
    sync = LangSmithDatasetSync(client)
    sync.sync(DATASET, [_example("keep"), _example("edit"), _example("drop")])
    client.writes.clear()
    sync.sync(
        DATASET, [_example("keep"), _example("edit", text="New"), _example("add")]
    )
    assert sorted(client.writes) == [
        ("create_examples", 1),
        ("delete_examples", 1),
        ("update_examples", 1),
    ]
    assert client.deleted == {_id_of("drop")}
    assert client.stored_texts(DATASET) == {
        "add": "Hello",
        "edit": "New",
        "keep": "Hello",
    }


def test_sync_batches_updates_and_deletes_but_not_creates() -> None:
    """Creates go in one call; updates and deletes go at most 100 at a time."""
    client = FakeLangSmithClient()
    sync = LangSmithDatasetSync(client)
    case_ids = [f"case-{index}" for index in range(250)]
    sync.sync(DATASET, [_example(case_id) for case_id in case_ids])
    sync.sync(DATASET, [_example(case_id, text="Edited") for case_id in case_ids])
    sync.sync(DATASET, [])
    assert client.writes == [
        ("create_dataset", 1),
        ("create_examples", 250),
        ("update_examples", 100),
        ("update_examples", 100),
        ("update_examples", 50),
        ("delete_examples", 100),
        ("delete_examples", 100),
        ("delete_examples", 50),
    ]


def test_re_adding_a_deleted_case_conflicts_on_its_reused_id() -> None:
    """A removed case that comes back reuses its soft-deleted ID, so creating fails."""
    client = FakeLangSmithClient()
    sync = LangSmithDatasetSync(client)
    sync.sync(DATASET, [_example("a"), _example("b")])
    sync.sync(DATASET, [_example("a")])
    with pytest.raises(LangSmithConflictError):
        sync.sync(DATASET, [_example("a"), _example("b")])
