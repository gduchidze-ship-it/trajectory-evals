"""Pins the outputs LangSmith hands evaluators when a target raises."""

import uuid
from typing import Any

import pytest
from langsmith import Client, aevaluate
from langsmith.schemas import Example

from trajectory_evals.suites.jailbreak.utils import _target_errored


async def _target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Raise for flagged inputs and answer the rest."""
    if inputs["fail"]:
        raise RuntimeError("target failed")
    return {"is_jailbreak": False, "decided_by": "screen"}


@pytest.mark.asyncio
async def test_aevaluate_passes_outputs_that_target_errored_recognises() -> None:
    """A raised target reaches evaluators as outputs that _target_errored flags."""
    seen: list[tuple[bool, dict[str, Any]]] = []

    def record(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        """Record each row's flag and outputs."""
        seen.append((inputs["fail"], outputs))
        return {"key": "recorded", "score": True}

    dataset_id = uuid.uuid4()
    examples = [
        Example(id=uuid.uuid4(), dataset_id=dataset_id, inputs={"fail": fail})
        for fail in (False, True)
    ]
    offline = Client(
        api_key="offline", auto_batch_tracing=False, tracing_sampling_rate=0.0
    )
    await aevaluate(
        _target,
        data=examples,
        evaluators=[record],
        upload_results=False,
        client=offline,
    )
    assert sorted((fail, _target_errored(out)) for fail, out in seen) == [
        (False, False),
        (True, True),
    ]
