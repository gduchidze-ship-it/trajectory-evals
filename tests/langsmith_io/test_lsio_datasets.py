"""Tests for pushing golden JSONL to a LangSmith dataset and pulling it back."""

from pathlib import Path
from uuid import uuid4

import pytest

from tests.langsmith_io.lsio_fakes import (
    GOLDEN_ROWS,
    FakeClient,
    read_jsonl,
    write_jsonl,
)
from trajectory_evals.core.dataset import SyncPlan, build_payloads, plan_sync
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.langsmith_io import _output
from trajectory_evals.langsmith_io.datasets import pull_dataset, push_dataset
from trajectory_evals.langsmith_io.settings import settings

DATASET = "support/golden"
NAMESPACE = settings.defaults.namespace


def _pushed(tmp_path: Path) -> FakeClient:
    """Return a client whose DATASET holds GOLDEN_ROWS pushed from golden.jsonl."""
    client = FakeClient()
    source = write_jsonl(tmp_path / "golden.jsonl", GOLDEN_ROWS)
    assert push_dataset(client, source, DATASET, namespace=NAMESPACE) == DATASET
    return client


def test_push_refuses_a_case_still_flagged_for_review(tmp_path: Path) -> None:
    """A harvested row is pushed only once review clears its flag, never before."""
    client = FakeClient()
    reviewed = {**GOLDEN_ROWS[0], "review_required": False}
    flagged = {
        "id": "run-1",
        "messages": [{"role": "user", "content": "Hold the CR-V."}],
        "expected": {},
        "review_required": True,
    }
    harvested = write_jsonl(tmp_path / "harvested.jsonl", [reviewed, flagged])

    with pytest.raises(ValueError, match="'run-1' still has review_required set"):
        push_dataset(client, harvested, DATASET, namespace=NAMESPACE)
    assert (client.datasets, client.writes) == ({}, [])

    cleared = write_jsonl(tmp_path / "cleared.jsonl", [reviewed])
    push_dataset(client, cleared, DATASET, namespace=NAMESPACE)
    assert client.writes == [("create_dataset", 1), ("create_examples", 1)]


def test_pulled_rows_are_the_golden_rows_and_push_back_unchanged(
    tmp_path: Path,
) -> None:
    """A pull writes each case's row; pushing it back plans and writes nothing."""
    client = _pushed(tmp_path)
    writes = list(client.writes)
    out = tmp_path / "pulled" / "golden.jsonl"

    pulled = pull_dataset(client, DATASET, out=out, allow_tracked=True)

    assert pulled == out
    assert read_jsonl(out) == [
        {
            "id": "price-match",
            "messages": [{"role": "user", "content": "Do you price match?"}],
            "context": {},
            "expected": {"contains": ["price match"]},
            "risk": "low",
        },
        GOLDEN_ROWS[1],
    ]
    dataset_id = client.datasets[DATASET].id
    stored = {
        example.id: (example.metadata or {})["content_hash"]
        for example in client.list_examples(dataset_id=dataset_id)
    }
    examples = JsonlGoldenSource(out).examples()
    payloads = build_payloads(DATASET, examples, namespace=NAMESPACE)
    assert plan_sync(payloads, stored) == SyncPlan((), (), ())
    push_dataset(client, out, DATASET, namespace=NAMESPACE)
    assert client.writes == writes


def test_pull_names_a_row_by_example_id_without_case_id(tmp_path: Path) -> None:
    """An example pushed by other tools keeps its metadata and its own id."""
    client = FakeClient()
    dataset = client.create_dataset("foreign")
    example_id = uuid4()
    messages = [{"role": "user", "content": "Hi"}]
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "id": example_id,
                "inputs": {"messages": messages},
                "outputs": {},
                "metadata": {"owner": "qa", "dataset_split": ["base"]},
            }
        ],
    )
    out = tmp_path / "foreign.jsonl"

    pull_dataset(client, "foreign", out=out, as_of="prod", allow_tracked=True)

    assert read_jsonl(out) == [
        {
            "id": str(example_id),
            "messages": messages,
            "context": {},
            "expected": {},
            "owner": "qa",
        }
    ]
    assert client.as_of == ["prod"]


def test_pull_refuses_an_existing_file_before_any_call(tmp_path: Path) -> None:
    """An existing out is never overwritten, and LangSmith is not asked."""
    client = _pushed(tmp_path)
    listed = list(client.as_of)
    out = write_jsonl(tmp_path / "kept.jsonl", [{"id": "mine"}])

    with pytest.raises(FileExistsError, match="kept.jsonl"):
        pull_dataset(client, DATASET, out=out, allow_tracked=True)

    assert read_jsonl(out) == [{"id": "mine"}]
    assert client.as_of == listed


def test_pull_writes_under_data_only_unless_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside the git-ignored data folder a pull refuses unless allow_tracked."""
    client = _pushed(tmp_path)
    monkeypatch.setattr(_output, "DATA_DIR", tmp_path / "data")

    with pytest.raises(ValueError, match="allow_tracked"):
        pull_dataset(client, DATASET, out=tmp_path / "tracked.jsonl")
    pulled = pull_dataset(client, DATASET, out=tmp_path / "data" / "ls" / "g.jsonl")

    assert not (tmp_path / "tracked.jsonl").exists()
    assert [row["id"] for row in read_jsonl(pulled)] == ["price-match", "refund"]
