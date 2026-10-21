"""A real push and pull of a golden dataset through LangSmith; needs an API key."""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from langsmith import Client

from tests.langsmith_io.lsio_fakes import GOLDEN_ROWS, read_jsonl, write_jsonl
from trajectory_evals.core.dataset import SyncPlan, build_payloads, plan_sync
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.langsmith_io.datasets import pull_dataset, push_dataset
from trajectory_evals.langsmith_io.protocols import LangSmithClient
from trajectory_evals.langsmith_io.settings import settings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("LANGSMITH_API_KEY"), reason="needs LANGSMITH_API_KEY"
    ),
]


def test_a_pushed_dataset_pulls_back_and_pushes_unchanged(tmp_path: Path) -> None:
    """Push a throwaway dataset, pull it, and plan no writes for the pulled file."""
    client: LangSmithClient = Client()
    name = f"trajectory-evals-e2e-{uuid4().hex[:8]}"
    namespace = settings.defaults.namespace
    source = write_jsonl(tmp_path / "golden.jsonl", GOLDEN_ROWS)
    out = tmp_path / "pulled" / "golden.jsonl"
    try:
        push_dataset(client, source, name, namespace=namespace)
        pull_dataset(client, name, out=out, allow_tracked=True)

        pulled = sorted(read_jsonl(out), key=lambda row: row["id"])
        assert [row["id"] for row in pulled] == ["price-match", "refund"]
        assert pulled[1] == GOLDEN_ROWS[1]
        dataset = client.read_dataset(dataset_name=name)
        stored = {
            example.id: (example.metadata or {}).get("content_hash")
            for example in client.list_examples(dataset_id=dataset.id)
        }
        examples = JsonlGoldenSource(out).examples()
        payloads = build_payloads(name, examples, namespace=namespace)
        assert plan_sync(payloads, stored) == SyncPlan((), (), ())
    finally:
        # A push that failed before creating the dataset leaves nothing to delete.
        cleanup = Client()
        if cleanup.has_dataset(dataset_name=name):
            cleanup.delete_dataset(dataset_name=name)
