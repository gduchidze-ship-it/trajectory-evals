"""Golden JSONL files mirrored into LangSmith datasets, and datasets pulled back."""

from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from langsmith.schemas import Example

from trajectory_evals.core.dataset import LangSmithDatasetSync
from trajectory_evals.core.settings import settings as core_settings
from trajectory_evals.core.types import DatasetExample
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.golden.settings import settings as golden_settings
from trajectory_evals.langsmith_io._output import check_out, write_rows
from trajectory_evals.langsmith_io.harvest import REVIEW_FLAG
from trajectory_evals.langsmith_io.protocols import LangSmithClient
from trajectory_evals.langsmith_io.settings import settings


def push_dataset(
    client: LangSmithClient, path: Path, name: str, *, namespace: str
) -> str:
    """Mirror the golden JSONL file at path into the named dataset; return the name.

    Only changed examples are written, so pushing an unchanged file writes
    nothing. Raises CaseFormatError, naming path:line, for a bad row, and
    ValueError naming the flag for a case that still sets review_required, as
    every harvested row does until a human reviews it; both before any write.
    """
    examples = _reviewed(JsonlGoldenSource(path).examples())
    return LangSmithDatasetSync(client, namespace=namespace).sync(name, examples)


def _reviewed(examples: Iterable[DatasetExample]) -> Iterator[DatasetExample]:
    """Yield the examples, raising ValueError at one still flagged for review."""
    for example in examples:
        if example["metadata"].get(REVIEW_FLAG):
            message = settings.texts.under_review
            raise ValueError(
                message.format(case_id=example["case_id"], flag=REVIEW_FLAG)
            )
        yield example


def pull_dataset(
    client: LangSmithClient,
    name: str,
    *,
    out: Path,
    as_of: datetime | str | None = None,
    allow_tracked: bool = False,
) -> Path:
    """Write the named dataset, as of a version time or tag, as golden JSONL at out.

    Each row's id is the example's case_id metadata, else the example id. Its
    messages and context come from the inputs, expected from the outputs, and
    its other metadata becomes top-level keys, except those push writes and
    those LangSmith adds. LangSmith's own splits (dataset_split) are not carried
    over, so a foreign dataset's splits are lost; a row's split key round-trips.
    Pushing the file back under its original file name changes nothing, since
    push records the file name on every example.
    Refuses an existing out, and one outside data/ unless allow_tracked, before
    any call; raises LangSmithNotFoundError for an unknown dataset.
    """
    check_out(out, allow_tracked=allow_tracked)
    dataset = client.read_dataset(dataset_name=name)
    examples = client.list_examples(dataset_id=dataset.id, as_of=as_of)
    return write_rows(out, [_golden_row(example) for example in examples])


def _golden_row(example: Example) -> dict[str, Any]:
    """Return the golden JSONL row an example was pushed from."""
    metadata = dict(example.metadata or {})
    inputs = example.inputs or {}
    case_id = metadata.get(core_settings.data.case_id_key) or str(example.id)
    return {
        **{key: value for key, value in metadata.items() if key not in _dropped()},
        "id": case_id,
        "messages": inputs.get("messages"),
        "context": inputs.get("context", {}),
        "expected": example.outputs or {},
    }


def _dropped() -> set[str]:
    """Return the metadata keys push writes and LangSmith adds, never row keys."""
    core, golden = core_settings.data, golden_settings.data
    return {
        core.case_id_key,
        core.content_hash_key,
        golden.source_file_key,
        *settings.data.server_keys,
    }


__all__ = ["pull_dataset", "push_dataset"]
