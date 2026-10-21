"""Tests for scoring production runs online and building the agent's evaluators."""

import logging
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_DNS, uuid5

import pytest
from langsmith.run_trees import get_cached_client
from langsmith.schemas import TracerSession
from langsmith.utils import LangSmithNotFoundError, LangSmithRateLimitError

from tests.fakes import judge_reply, scripted
from tests.golden.golden_fakes import ModelFactory, agent_config
from tests.langsmith_io.lsio_fakes import (
    PROJECT,
    PROJECT_ID,
    SINCE,
    FakeClient,
    make_run,
)
from trajectory_evals.core.types import EvaluatorResult, SimpleAsyncEvaluator
from trajectory_evals.langsmith_io.score import (
    KeyCounts,
    ScoreReport,
    feedback_id,
    online_evaluators,
    score_runs,
)
from trajectory_evals.langsmith_io.settings import settings
from trajectory_evals.simulation.settings import settings as simulation_settings

USER = {"role": "user", "content": "Can you hold the CR-V for me?"}
REPLY = {"role": "assistant", "content": "I have put it on hold until Friday."}
INPUTS = {"messages": [USER]}
OUTPUTS = {"messages": [USER, REPLY]}


class ScriptedEvaluator:
    """An evaluator under one key that records its calls and gives one score."""

    def __init__(
        self, key: str, score: float | bool | None = True, *, fails: bool = False
    ) -> None:
        """Report score under key, or raise on every call when fails."""
        self.__name__ = key
        self.calls: list[dict[str, Any]] = []
        self._score = score
        self._fails = fails

    async def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any,
        reference_outputs: Any = None,
        **_: Any,
    ) -> EvaluatorResult:
        """Record the arguments, then score or fail."""
        self.calls.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "reference_outputs": reference_outputs,
            }
        )
        if self._fails:
            raise RuntimeError("judge unavailable")
        return EvaluatorResult(
            key=self.__name__,
            score=self._score,  # type: ignore[typeddict-item]
            comment="because",
            metadata=None,
            source_run_id=None,
        )


async def _score(
    client: FakeClient, evaluators: list[SimpleAsyncEvaluator], **options: Any
) -> ScoreReport:
    """Score PROJECT's runs with evaluators."""
    return await score_runs(
        client, PROJECT, evaluators, since=SINCE, limit=10, **options
    )


@pytest.mark.asyncio
async def test_feedback_names_the_run_project_trace_and_a_stable_id(
    network_attempts: list[Any],
) -> None:
    """Each score is written with session_id, trace_id, a uuid5 id, no retention."""
    run = make_run(INPUTS, OUTPUTS)
    client = FakeClient([run])
    polite = ScriptedEvaluator("polite")

    report = await _score(client, [polite])

    namespace = uuid5(NAMESPACE_DNS, settings.keys.feedback_namespace)
    assert client.feedback == [
        {
            "run_id": run.id,
            "key": "polite",
            "score": True,
            "comment": "because",
            "session_id": PROJECT_ID,
            "trace_id": run.trace_id,
            "feedback_id": uuid5(namespace, f"{run.id}:polite"),
            "start_time": SINCE,
            "feedback_source_type": "model",
            "extend_trace_retention": False,
        }
    ]
    assert polite.calls == [
        {
            "inputs": INPUTS,
            "outputs": {"text": REPLY["content"], "messages": [USER, REPLY]},
            "reference_outputs": {},
        }
    ]
    assert report == ScoreReport(
        runs=1, keys={"polite": KeyCounts(scored=1, written=1)}
    )
    assert client.flushes == 1
    assert network_attempts == []


@pytest.mark.asyncio
async def test_feedback_ids_repeat_on_a_rerun_and_differ_by_run_and_key() -> None:
    """Scoring the same runs twice gives the same ids; each run and key its own."""
    runs = [make_run(INPUTS, OUTPUTS), make_run(INPUTS, OUTPUTS)]
    first, second = FakeClient(runs), FakeClient(runs)
    keys = ["polite", "helpful"]

    await _score(first, [ScriptedEvaluator(key) for key in keys])
    await _score(second, [ScriptedEvaluator(key) for key in keys])

    ids = [row["feedback_id"] for row in first.feedback]
    assert ids == [row["feedback_id"] for row in second.feedback]
    assert len(set(ids)) == 4
    assert ids[0] == feedback_id(runs[0].id, "polite")


@pytest.mark.asyncio
async def test_runs_already_carrying_a_key_are_skipped_for_that_key() -> None:
    """A run with polite feedback is not graded for polite again, only for safe."""
    scored = make_run(INPUTS, OUTPUTS, feedback={"polite": 1.0})
    fresh = make_run(INPUTS, OUTPUTS)
    polite, safe = ScriptedEvaluator("polite"), ScriptedEvaluator("safe")

    report = await _score(FakeClient([scored, fresh]), [polite, safe])

    assert (len(polite.calls), len(safe.calls)) == (1, 2)
    assert report.keys == {
        "polite": KeyCounts(scored=1, skipped=1, written=1),
        "safe": KeyCounts(scored=2, written=2),
    }


@pytest.mark.asyncio
async def test_a_dry_run_logs_scores_and_writes_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Even a sampling client may dry-run: it grades, logs, and writes nothing."""
    run = make_run(INPUTS, OUTPUTS)
    client = FakeClient([run], tracing_sample_rate=0.0)
    caplog.set_level(logging.INFO, logger="trajectory_evals.langsmith_io.score")

    report = await _score(client, [ScriptedEvaluator("polite")], dry_run=True)

    assert client.feedback == []
    assert report.keys == {"polite": KeyCounts(scored=1)}
    assert f"would write polite=True on run {run.id}" in caplog.text


@pytest.mark.asyncio
async def test_failing_or_unscored_evaluators_are_counted_not_fatal() -> None:
    """A raising evaluator and a None score are errors; the others still write."""
    runs = [make_run(INPUTS, OUTPUTS), make_run(INPUTS, OUTPUTS)]
    client = FakeClient(runs)
    evaluators = [
        ScriptedEvaluator("broken", fails=True),
        ScriptedEvaluator("unsure", None),
        ScriptedEvaluator("polite"),
    ]

    report = await _score(client, list(evaluators))

    assert report.keys == {
        "broken": KeyCounts(errors=2),
        "unsure": KeyCounts(errors=2),
        "polite": KeyCounts(scored=2, written=2),
    }
    assert [row["key"] for row in client.feedback] == ["polite", "polite"]


@pytest.mark.asyncio
async def test_runs_without_a_conversation_or_reply_are_skipped() -> None:
    """A run still pending, errored, or without a user message is not graded."""
    runs = [
        make_run(INPUTS, None),
        make_run(INPUTS, {"messages": [USER]}),
        make_run(INPUTS, {"messages": REPLY["content"]}),
        make_run({"messages": [REPLY]}, OUTPUTS),
    ]
    polite = ScriptedEvaluator("polite")

    report = await _score(FakeClient(runs), [polite])

    assert polite.calls == []
    assert report == ScoreReport(runs=4, keys={"polite": KeyCounts(skipped=4)})


@pytest.mark.asyncio
async def test_a_sampling_client_is_refused_before_any_call() -> None:
    """A client that samples feedback would drop scores, so it is refused."""
    client = FakeClient([make_run(INPUTS, OUTPUTS)], tracing_sample_rate=0.5)

    with pytest.raises(ValueError, match="0.5"):
        await _score(client, [ScriptedEvaluator("polite")])

    assert (client.projects, client.feedback) == ([], [])


@pytest.mark.asyncio
async def test_a_limit_below_one_is_refused_before_any_call() -> None:
    """limit=0 is a ValueError before the project is read or the client flushed."""
    client = FakeClient([make_run(INPUTS, OUTPUTS)])

    with pytest.raises(ValueError, match="limit"):
        await score_runs(
            client, PROJECT, [ScriptedEvaluator("polite")], since=SINCE, limit=0
        )

    assert (client.projects, client.flushes) == ([], 0)


@pytest.mark.asyncio
async def test_only_runs_that_did_not_error_are_read() -> None:
    """Scoring asks for has_error=False, alongside any filter it is given."""
    client = FakeClient()

    await _score(client, [ScriptedEvaluator("polite")], filter='has(tags, "prod")')

    [query] = client.runs.queries
    assert (query["has_error"], query["filter"]) == (False, 'has(tags, "prod")')


class _RefusingFeedback(FakeClient):
    """A client whose LangSmith refuses every polite write, as on a 429."""

    def create_feedback(self, run_id: str | None, key: str, **fields: Any) -> None:
        """Refuse polite feedback; record any other."""
        if key == "polite":
            raise LangSmithRateLimitError("Rate limit exceeded")
        super().create_feedback(run_id, key, **fields)


@pytest.mark.asyncio
async def test_a_refused_write_is_counted_and_scoring_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A LangSmithError on one write is an error for its key, logged by type only."""
    client = _RefusingFeedback([make_run(INPUTS, OUTPUTS), make_run(INPUTS, OUTPUTS)])

    report = await _score(
        client, [ScriptedEvaluator("polite"), ScriptedEvaluator("safe")]
    )

    assert report.keys == {
        "polite": KeyCounts(scored=2, errors=2),
        "safe": KeyCounts(scored=2, written=2),
    }
    assert "LangSmithRateLimitError" in caplog.text
    assert "Rate limit exceeded" not in caplog.text


class _NoProject(FakeClient):
    """A client whose project lookup fails, as for a mistyped project name."""

    async def aread_project(self, *, project_name: str) -> TracerSession:
        """Fail as LangSmith does for an unknown project."""
        raise LangSmithNotFoundError(f"Project {project_name} not found")


@pytest.mark.asyncio
async def test_the_client_is_flushed_even_when_reading_fails() -> None:
    """flush runs on the way out of a failed score, so nothing queued is lost."""
    client = _NoProject()

    with pytest.raises(LangSmithNotFoundError):
        await _score(client, [ScriptedEvaluator("polite")])

    assert client.flushes == 1


@pytest.mark.asyncio
async def test_online_evaluators_are_reference_free_judges_and_the_policy_judge(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A judge needing references is left out; the policy judge grades the reply.

    Tracing is on, yet no judge call is traced anywhere.
    """
    (tmp_path / "correct.md").write_text(
        "{outputs} against {reference_outputs}", encoding="utf-8"
    )
    (tmp_path / "policy.md").write_text("Never hold a car.", encoding="utf-8")
    polite_judge, policy_judge = (
        scripted(judge_reply(True)),
        scripted(judge_reply(True)),
    )
    redteam_model = simulation_settings.defaults.judge_model
    factory = ModelFactory({"openai:polite": polite_judge, redteam_model: policy_judge})
    judges = {
        "polite": {
            "prompt": "openevals:TASK_COMPLETION_PROMPT",
            "model": "openai:polite",
            "reference_free": True,
        },
        "correct": {"prompt": "correct.md", "model": "openai:correct"},
    }
    config = agent_config(tmp_path, judges=judges, redteam={"policy": "policy.md"})
    client = FakeClient([run := make_run(INPUTS, OUTPUTS)])

    evaluators = online_evaluators(config, model_factory=factory)
    report = await _score(client, evaluators)

    assert [getattr(e, "__name__") for e in evaluators] == [
        "polite",
        "policy_violation",
    ]
    assert factory.names == ["openai:polite", redteam_model]
    assert [(row["key"], row["score"]) for row in client.feedback] == [
        ("polite", True),
        ("policy_violation", True),
    ]
    assert report.runs == 1 and client.feedback[1]["run_id"] == run.id
    prompt = policy_judge.prompts[0][-1].text
    assert "Never hold a car." in prompt and REPLY["content"] in prompt
    # Sends anything the tracer queued, so a traced judge shows up as an attempt.
    get_cached_client().flush(timeout=5)
    assert network_attempts == []


def test_without_a_policy_there_is_no_policy_judge(tmp_path: Path) -> None:
    """An agent with no [redteam] policy and no reference-free judge gets none."""
    config = agent_config(tmp_path, redteam={"attempts": 2})

    assert online_evaluators(config, model_factory=ModelFactory()) == []
