"""Online evals: reference-free evaluators over production runs, as feedback."""

import asyncio
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import NAMESPACE_DNS, UUID, uuid5

from langchain.chat_models import init_chat_model
from langsmith import tracing_context
from langsmith.types import Run, RunSelectField
from langsmith.utils import LangSmithError

from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.evaluators import _async_evaluator
from trajectory_evals.core.types import AgentOutputs, EvaluatorResult
from trajectory_evals.core.types import SimpleAsyncEvaluator as Evaluator
from trajectory_evals.golden.graders import ModelFactory, reference_free_judges
from trajectory_evals.langsmith_io._runs import (
    Conversation,
    check_limit,
    query_root_runs,
    read_conversation,
)
from trajectory_evals.langsmith_io.protocols import LangSmithClient
from trajectory_evals.langsmith_io.settings import settings
from trajectory_evals.simulation.judge import AttackJudge
from trajectory_evals.simulation.settings import RedteamSection, agent_table
from trajectory_evals.simulation.settings import settings as simulation_settings

logger = logging.getLogger(__name__)

SELECTS: tuple[RunSelectField, ...] = (
    "ID",
    "TRACE_ID",
    "PROJECT_ID",
    "START_TIME",
    "INPUTS",
    "OUTPUTS",
    "FEEDBACK_STATS",
)
MODEL_SOURCE = "model"


@dataclass(frozen=True)
class KeyCounts:
    """What scoring did for one feedback key over the runs read.

    skipped counts runs not evaluated: already carrying the key, or with no
    conversation and reply to grade. errors counts evaluator calls that raised
    or gave no score, and writes LangSmith refused. written counts feedback
    LangSmith accepted: each write is posted and awaited, never queued. It is 0
    on a dry run.
    """

    scored: int = 0
    skipped: int = 0
    errors: int = 0
    written: int = 0


@dataclass(frozen=True)
class ScoreReport:
    """How many root runs were read, and what scoring did per feedback key."""

    runs: int
    keys: Mapping[str, KeyCounts]


@dataclass
class _Scoring:
    """One score_runs call: its client, mode and evaluators, and the counts so far."""

    client: LangSmithClient
    dry_run: bool
    evaluators: Sequence[Evaluator]
    read: int = 0
    unreadable: int = 0
    tallies: defaultdict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    async def score(self, run: Run, conversation: Conversation | None) -> None:
        """Run every evaluator on one run, or skip it if it has nothing to grade."""
        self.read += 1
        if conversation is None or not conversation.reply:
            self.unreadable += 1
            for evaluator in self.evaluators:
                self.tallies[_key(evaluator)]["skipped"] += 1
            return
        # shortcut: evaluators run one at a time; gather them if scoring is slow.
        for evaluator in self.evaluators:
            await self._evaluate(run, conversation, evaluator)

    def report(self) -> ScoreReport:
        """Return the counts of every evaluator's key and of each key a result used."""
        keys = dict.fromkeys([*map(_key, self.evaluators), *self.tallies])
        counts = {key: KeyCounts(**self.tallies[key]) for key in keys}
        return ScoreReport(runs=self.read, keys=counts)

    async def _evaluate(
        self, run: Run, conversation: Conversation, evaluator: Evaluator
    ) -> None:
        """Run evaluator on the run unless it carries its key, and record results."""
        key = _key(evaluator)
        if key in (run.feedback_stats or {}):
            self.tallies[key]["skipped"] += 1
            return
        transcript = [*conversation.messages, *conversation.reply]
        try:
            # Judges read unmasked production text: never trace it to any project.
            with tracing_context(enabled=False):
                result = await evaluator(
                    inputs={"messages": list(conversation.messages)},
                    outputs=AgentOutputs(text=conversation.text, messages=transcript),
                    reference_outputs={},
                )
        # One evaluator failing on one run must not stop the batch; errors shows it.
        except Exception as error:
            texts = settings.texts
            logger.warning(texts.evaluator_failed, key, run.id, type(error).__name__)
            self.tallies[key]["errors"] += 1
            return
        for item in result if isinstance(result, list) else [result]:
            await self._record(run, item)

    async def _record(self, run: Run, result: EvaluatorResult) -> None:
        """Count one result and write it as feedback, unless unscored or dry."""
        key, score = result["key"], result["score"]
        if score is None:
            self.tallies[key]["errors"] += 1
            return
        self.tallies[key]["scored"] += 1
        if self.dry_run:
            logger.info(settings.texts.dry_run, key, score, run.id)
            return
        try:
            await asyncio.to_thread(self._write, run, result)
        # One refused write must not stop the batch; errors shows it.
        except LangSmithError as error:
            texts = settings.texts
            logger.warning(texts.feedback_failed, key, run.id, type(error).__name__)
            self.tallies[key]["errors"] += 1
            return
        self.tallies[key]["written"] += 1

    def _write(self, run: Run, result: EvaluatorResult) -> None:
        """Post one score as feedback on the run and wait for LangSmith to take it."""
        key = result["key"]
        self.client.create_feedback(
            run.id,
            key,
            score=result["score"],
            comment=result.get("comment"),
            session_id=run.project_id,
            trace_id=run.trace_id,
            feedback_id=feedback_id(run.id, key),
            start_time=run.start_time,
            feedback_source_type=MODEL_SOURCE,
            # Scoring must not move production traces to paid, longer-kept storage.
            extend_trace_retention=False,
        )


def feedback_id(run_id: str | None, key: str) -> UUID:
    """Return the id of key's feedback on the run, the same on every rerun."""
    namespace = uuid5(NAMESPACE_DNS, settings.keys.feedback_namespace)
    return uuid5(namespace, f"{run_id}:{key}")


async def score_runs(
    client: LangSmithClient,
    project: str,
    evaluators: Sequence[Evaluator],
    *,
    since: datetime,
    limit: int,
    filter: str | None = None,
    inputs_key: str = settings.defaults.messages_key,
    outputs_key: str = settings.defaults.messages_key,
    dry_run: bool = False,
) -> ScoreReport:
    """Grade up to limit root runs of project since since, and write the scores.

    Only runs that did not error are read. limit caps the runs read, already
    scored ones included, in the order LangSmith returns them, which the SDK
    does not document; to backfill, narrow since rather than raise limit.

    Each evaluator is called with inputs {"messages"}, the AgentOutputs of the
    run's conversation and reply, and reference_outputs {}, with tracing off,
    and is keyed by its __name__, as core's wrappers set it; a run already
    carrying that key is skipped. Each scored result becomes feedback with the
    run's project, trace and a deterministic id; dry_run logs it instead and
    writes nothing. Scoring never extends production trace retention, for cost
    and PII: each write passes extend_trace_retention=False, which makes it a
    blocking POST, and a user who wants extended retention must change that
    deliberately. A write LangSmith refuses (a LangSmithError) is logged by
    type and counted in errors. The client is flushed before returning, even
    on error. Raises ValueError, before any call, for a limit below 1 or a
    client that samples feedback unless dry_run; raises LangSmithNotFoundError
    for an unknown project and a langsmith.APIError subclass when the query
    fails.
    """
    check_limit(limit)
    if not dry_run:
        _require_unsampled(client)
    conditions: dict[str, Any] = {"has_error": False}
    if filter is not None:
        conditions["filter"] = filter
    runs = query_root_runs(
        client, project, since=since, limit=limit, selects=SELECTS, **conditions
    )
    scoring = _Scoring(client, dry_run, evaluators)
    try:
        async for run in runs:
            conversation = read_conversation(
                run, inputs_key=inputs_key, outputs_key=outputs_key
            )
            await scoring.score(run, conversation)
    finally:
        await asyncio.to_thread(client.flush)
    read, unreadable = scoring.read, scoring.unreadable
    logger.info(
        settings.texts.scored, read, project, unreadable, inputs_key, outputs_key
    )
    return scoring.report()


def _key(evaluator: Evaluator) -> str:
    """Return the feedback key an evaluator reports under: its __name__."""
    return getattr(evaluator, "__name__", type(evaluator).__name__)


def _require_unsampled(client: LangSmithClient) -> None:
    """Refuse a client that would silently drop a share of the feedback."""
    rate = client.tracing_sample_rate
    if rate is not None and rate < 1:
        raise ValueError(settings.texts.sampled_client.format(rate=rate))


def online_evaluators(
    config: AgentConfig, *, model_factory: ModelFactory = init_chat_model
) -> list[Evaluator]:
    """Return the agent's evaluators that need no reference outputs.

    These are each [judges.<name>] with reference_free = true, then, when
    [redteam] sets a policy, policy_violation: the attack judge of whether the
    reply after the last user message violates that policy, scored 1 if so.
    model_factory makes each judge's chat model. Raises AgentConfigError for a
    bad table or a file it cannot read.
    """
    evaluators = reference_free_judges(config, model_factory=model_factory)
    redteam = agent_table(config, "redteam", RedteamSection)
    if redteam.policy is not None:
        policy = config.read_text(redteam.policy)
        judge = AttackJudge(model_factory(redteam.judge_model), policy)
        evaluators.append(_policy_violation(judge))
    return evaluators


def _policy_violation(judge: AttackJudge) -> Evaluator:
    """Return the evaluator scoring whether the latest reply breaks the policy."""

    async def score(*, outputs: AgentOutputs, **_: Any) -> bool:
        """Return whether the attack judge finds the latest reply in violation."""
        return await judge(outputs["messages"])

    return _async_evaluator(
        score,
        settings.keys.policy_violation,
        ls_framework=simulation_settings.keys.ls_framework,
    )


__all__ = [
    "MODEL_SOURCE",
    "SELECTS",
    "KeyCounts",
    "ScoreReport",
    "feedback_id",
    "online_evaluators",
    "score_runs",
]
