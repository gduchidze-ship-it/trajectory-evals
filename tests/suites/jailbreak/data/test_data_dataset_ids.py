"""Tests that the suite's dataset mirror keeps the ids and keys it always wrote."""

from typing import Any, cast
from uuid import UUID

from trajectory_evals.core.settings import settings as core_settings
from trajectory_evals.suites.jailbreak.data import LangSmithDatasetSync
from trajectory_evals.suites.jailbreak.settings import settings

PINNED_ID = UUID("3f031a4d-b831-5463-94aa-a7b6595fcb69")


def test_suite_sync_stores_a_case_under_its_old_id() -> None:
    """The suite's namespace still gives a case the id LangSmith already stores."""
    sync = LangSmithDatasetSync(cast(Any, None))

    assert sync.example_id("jailbreak-detection/prod_sample", "row-1") == PINNED_ID


def test_core_writes_the_metadata_keys_the_suite_reserves() -> None:
    """Rows may not use case_id or content_hash, the keys core writes per example."""
    keys = ("case_id_key", "content_hash_key")

    assert {key: getattr(core_settings.data, key) for key in keys} == {
        key: getattr(settings.data, key) for key in keys
    }
