"""Tests for harvesting production root runs into golden-case candidates."""

import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.load import dumpd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tests.langsmith_io.lsio_fakes import (
    PROJECT,
    PROJECT_ID,
    SINCE,
    FakeClient,
    make_run,
    read_jsonl,
    write_jsonl,
)
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.langsmith_io import _output
from trajectory_evals.langsmith_io.harvest import SELECTS, harvest_runs

ASKED = {"role": "user", "content": "Hi"}
GREETED = {"role": "assistant", "content": "Hello! How can I help?"}
STOCK = {"role": "user", "content": "Is the CR-V in stock?"}
ANSWER = {"role": "assistant", "content": "Yes, two are on the lot."}
CONVERSATION = [ASKED, GREETED, STOCK]
FEEDBACK_BELOW = 'and(eq(feedback_key, "user_score"), lt(feedback_score, 0.5))'


async def _harvest(
    client: FakeClient, out: Path, *, limit: int = 10, **options: Any
) -> list[dict[str, Any]]:
    """Harvest PROJECT into out, wherever it is, and return the rows written."""
    await harvest_runs(
        client,
        PROJECT,
        out=out,
        since=SINCE,
        limit=limit,
        allow_tracked=True,
        **options,
    )
    return read_jsonl(out)


@pytest.mark.asyncio
async def test_each_run_becomes_a_review_required_golden_candidate(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A row keeps the conversation, the reply, the run's ids, feedback and error."""
    answered = make_run(
        {"messages": CONVERSATION, "client_key": "k"},
        {"messages": [*CONVERSATION, ANSWER]},
        feedback={"user_score": 0.0},
    )
    errored = make_run({"messages": [STOCK]}, None, error="TimeoutError: mcp")
    client = FakeClient([answered, errored])
    out = tmp_path / "candidates.jsonl"

    rows = await _harvest(client, out)

    assert rows == [
        {
            "id": f"run-{answered.id}",
            "messages": CONVERSATION,
            "expected": {},
            "review_required": True,
            "observed_output": ANSWER["content"],
            "source_run_id": answered.id,
            "trace_id": answered.trace_id,
            "project": PROJECT,
            "feedback": {"user_score": 0.0},
            "error": None,
        },
        {
            "id": f"run-{errored.id}",
            "messages": [STOCK],
            "expected": {},
            "review_required": True,
            "observed_output": "",
            "source_run_id": errored.id,
            "trace_id": errored.trace_id,
            "project": PROJECT,
            "feedback": {},
            "error": "TimeoutError: mcp",
        },
    ]
    [case, _] = JsonlGoldenSource(out).cases()
    assert case.metadata["review_required"] is True
    assert network_attempts == []


@pytest.mark.asyncio
async def test_langchain_serialized_state_at_a_dotted_path_is_normalized(
    tmp_path: Path,
) -> None:
    """dumpd and model_dump messages under a nested key come out as OpenAI dicts."""
    call = {"name": "check_stock", "args": {"model": "CR-V"}, "id": "call-1"}
    history = [SystemMessage("Be brief."), HumanMessage("Is the CR-V in stock?")]
    reply = [
        AIMessage("", tool_calls=[call]),
        ToolMessage("2 in stock", tool_call_id="call-1"),
        AIMessage("Yes, two are on the lot."),
    ]
    run = make_run(
        {"state": {"messages": [dumpd(message) for message in history]}},
        {"state": {"messages": [m.model_dump() for m in [*history, *reply]]}},
    )

    [row] = await _harvest(
        FakeClient([run]),
        tmp_path / "out.jsonl",
        inputs_key="state.messages",
        outputs_key="state.messages",
    )

    assert row["messages"] == [{"role": "system", "content": "Be brief."}, STOCK]
    assert row["observed_output"] == ANSWER["content"]


@pytest.mark.asyncio
async def test_runs_without_a_final_user_message_are_skipped_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A run ending with the agent, missing the key or not holding messages is out."""
    runs = [
        make_run({"messages": [ASKED, GREETED]}),
        make_run({"history": CONVERSATION}),
        make_run({"messages": {"client_key": "k"}}),
        kept := make_run({"messages": CONVERSATION}),
    ]
    caplog.set_level(logging.INFO, logger="trajectory_evals.langsmith_io.harvest")

    rows = await _harvest(FakeClient(runs), tmp_path / "out.jsonl")

    assert [row["source_run_id"] for row in rows] == [kept.id]
    assert "harvested 1 of 4 runs from dev-sales; 3 had no" in caplog.text


@pytest.mark.asyncio
async def test_the_query_always_bounds_the_start_time_and_selects_fields(
    tmp_path: Path,
) -> None:
    """Root runs of the resolved project id since since, with the needed fields."""
    client = FakeClient([make_run({"messages": CONVERSATION})])

    await _harvest(client, tmp_path / "out.jsonl", limit=5)

    assert client.projects == [PROJECT]
    assert client.runs.queries == [
        {
            "project_ids": [PROJECT_ID],
            "is_root": True,
            "min_start_time": SINCE,
            "selects": list(SELECTS),
            "page_size": 5,
        }
    ]


@pytest.mark.parametrize(
    ("options", "conditions"),
    [
        ({"only_errors": True}, {"has_error": True}),
        (
            {"feedback_key": "user_score", "feedback_below": 0.5},
            {"filter": FEEDBACK_BELOW},
        ),
        ({"feedback_key": "user_score"}, {"filter": 'eq(feedback_key, "user_score")'}),
        (
            {"filter": 'has(tags, "prod")', "only_errors": True, "feedback_key": "x"},
            {
                "filter": 'and(has(tags, "prod"), eq(feedback_key, "x"))',
                "has_error": True,
            },
        ),
    ],
    ids=["errors", "feedback-below", "feedback-key", "combined"],
)
@pytest.mark.asyncio
async def test_options_become_the_query_filter(
    tmp_path: Path, options: dict[str, Any], conditions: dict[str, Any]
) -> None:
    """Errors use has_error; feedback and a raw filter are ANDed in one filter."""
    client = FakeClient()

    await _harvest(client, tmp_path / "out.jsonl", **options)

    [query] = client.runs.queries
    assert {key: query[key] for key in ("filter", "has_error") if key in query} == (
        conditions
    )


@pytest.mark.asyncio
async def test_feedback_below_without_a_key_is_refused_before_any_call(
    tmp_path: Path,
) -> None:
    """A score bound with no feedback key to bound is a ValueError."""
    client = FakeClient()

    with pytest.raises(ValueError, match="feedback_key"):
        await _harvest(client, tmp_path / "out.jsonl", feedback_below=0.5)

    assert (client.projects, client.runs.queries) == ([], [])


@pytest.mark.asyncio
async def test_reading_stops_at_the_limit(tmp_path: Path) -> None:
    """No run past limit is asked for, and the page is no larger than limit."""
    client = FakeClient([make_run({"messages": CONVERSATION}) for _ in range(3)])

    rows = await _harvest(client, tmp_path / "out.jsonl", limit=2)

    assert (len(rows), client.runs.yielded) == (2, 2)
    assert client.runs.queries[0]["page_size"] == 2


@pytest.mark.asyncio
async def test_customer_data_is_written_only_under_data_unless_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside the git-ignored data folder harvest refuses, before any call."""
    monkeypatch.setattr(_output, "DATA_DIR", tmp_path / "data")
    client = FakeClient([make_run({"messages": CONVERSATION})])
    tracked = tmp_path / "tests" / "candidates.jsonl"

    with pytest.raises(ValueError, match="allow_tracked"):
        await harvest_runs(client, PROJECT, out=tracked, since=SINCE, limit=1)
    ignored = tmp_path / "data" / "langsmith" / "candidates.jsonl"
    await harvest_runs(client, PROJECT, out=ignored, since=SINCE, limit=1)

    assert not tracked.exists()
    assert len(read_jsonl(ignored)) == 1
    assert len(client.runs.queries) == 1


@pytest.mark.asyncio
async def test_an_existing_file_is_never_overwritten(tmp_path: Path) -> None:
    """A harvest into an existing file fails before any call and keeps the file."""
    client = FakeClient([make_run({"messages": CONVERSATION})])
    out = write_jsonl(tmp_path / "reviewed.jsonl", [{"id": "labelled"}])

    with pytest.raises(FileExistsError):
        await _harvest(client, out)

    assert read_jsonl(out) == [{"id": "labelled"}]
    assert client.runs.queries == []
