"""Production root runs harvested into golden-case candidates for a human to label."""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langsmith.types import Run, RunSelectField

from trajectory_evals.core.types import ChatCompletionMessage
from trajectory_evals.langsmith_io._output import DATA_DIR, check_out, write_rows
from trajectory_evals.langsmith_io._runs import (
    Conversation,
    check_limit,
    query_root_runs,
    read_conversation,
)
from trajectory_evals.langsmith_io.protocols import LangSmithClient
from trajectory_evals.langsmith_io.settings import settings

logger = logging.getLogger(__name__)

HARVEST_DIR = DATA_DIR / "langsmith"
# CandidateRow's flag; push_dataset refuses a case that still sets it.
REVIEW_FLAG = "review_required"
SELECTS: tuple[RunSelectField, ...] = (
    "ID",
    "TRACE_ID",
    "INPUTS",
    "OUTPUTS",
    "ERROR",
    "FEEDBACK_STATS",
)


class CandidateRow(TypedDict):
    """A golden row harvested from a root run; it grades nothing until labelled.

    expected is empty until a human fills it in. observed_output is the text of
    the agent's reply, feedback each feedback key's mean score on the run, and
    error the run's error; every key besides id, messages and expected is
    metadata to a golden source.
    """

    id: str
    messages: list[ChatCompletionMessage]
    expected: dict[str, Any]
    review_required: bool
    observed_output: str
    source_run_id: str | None
    trace_id: str | None
    project: str
    feedback: dict[str, float | None]
    error: str | None


async def harvest_runs(
    client: LangSmithClient,
    project: str,
    *,
    out: Path,
    since: datetime,
    limit: int,
    filter: str | None = None,
    only_errors: bool = False,
    feedback_key: str | None = None,
    feedback_below: float | None = None,
    inputs_key: str = settings.defaults.messages_key,
    outputs_key: str = settings.defaults.messages_key,
    allow_tracked: bool = False,
) -> Path:
    """Write a candidate golden row per readable root run of project to out.

    Reads up to limit root runs started since since that match filter, errored
    when only_errors, and, with feedback_key, carry that feedback, scored below
    feedback_below when set. A run whose inputs_key holds no conversation ending
    with a user message is skipped and counted in the log; a run with no reply
    gets observed_output "". Refuses a limit below 1, an existing out, one
    outside data/ unless allow_tracked, and a feedback_below that is not finite
    or has no feedback_key, before any call; raises LangSmithNotFoundError for
    an unknown project and a langsmith.APIError subclass when the query fails,
    writing no file.
    """
    check_limit(limit)
    check_out(out, allow_tracked=allow_tracked)
    conditions = _conditions(filter, only_errors, feedback_key, feedback_below)
    runs = query_root_runs(
        client, project, since=since, limit=limit, selects=SELECTS, **conditions
    )
    rows: list[CandidateRow] = []
    read = 0
    async for run in runs:
        read += 1
        conversation = read_conversation(
            run, inputs_key=inputs_key, outputs_key=outputs_key
        )
        if conversation is not None:
            rows.append(_candidate(run, conversation, project))
    skipped = read - len(rows)
    logger.info(settings.texts.harvested, len(rows), read, project, skipped, inputs_key)
    return write_rows(out, rows)


def _conditions(
    raw_filter: str | None,
    only_errors: bool,
    feedback_key: str | None,
    feedback_below: float | None,
) -> dict[str, Any]:
    """Return the runs.query filter and has_error arguments the options ask for."""
    texts = settings.texts
    if feedback_below is not None and not math.isfinite(feedback_below):
        raise ValueError(texts.below_not_finite.format(value=feedback_below))
    clauses = [] if raw_filter is None else [raw_filter]
    if feedback_key is not None:
        clauses.append(_feedback_clause(feedback_key, feedback_below))
    elif feedback_below is not None:
        raise ValueError(texts.below_needs_key)
    conditions: dict[str, Any] = {"has_error": True} if only_errors else {}
    if clauses:
        joined = ", ".join(clauses)
        conditions["filter"] = clauses[0] if len(clauses) == 1 else f"and({joined})"
    return conditions


def _feedback_clause(key: str, below: float | None) -> str:
    """Return the filter for runs with feedback key, scored below below if set."""
    # json.dumps quotes the key as LangSmith's own filters do.
    has_key = f"eq(feedback_key, {json.dumps(key)})"
    return has_key if below is None else f"and({has_key}, lt(feedback_score, {below}))"


def _candidate(run: Run, conversation: Conversation, project: str) -> CandidateRow:
    """Return the candidate row of one readable run."""
    return CandidateRow(
        id=f"run-{run.id}",
        messages=list(conversation.messages),
        expected={},
        review_required=True,
        observed_output=conversation.text,
        source_run_id=run.id,
        trace_id=run.trace_id,
        project=project,
        feedback={key: stats.avg for key, stats in (run.feedback_stats or {}).items()},
        error=run.error,
    )


__all__ = ["HARVEST_DIR", "REVIEW_FLAG", "SELECTS", "CandidateRow", "harvest_runs"]
