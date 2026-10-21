"""The reflective prompt optimizer: one lineage of candidates, hill-climbing on val."""

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.protocols import Agent, CaseSource
from trajectory_evals.core.runner import ExperimentRow, ExperimentRunner
from trajectory_evals.golden.graders import ModelFactory, _escape_tags
from trajectory_evals.optimize._evaluation import (
    BEST_DIR,
    CANDIDATES_DIR,
    VS_BASELINE,
    VS_PREVIOUS,
    Candidate,
    CandidateRecord,
    Evaluation,
    Run,
    Summary,
    at_most,
    deltas,
    floor_failures,
    pair,
    redteam_failures,
    start_run,
)
from trajectory_evals.optimize.settings import read_prompts, settings
from trajectory_evals.optimize.splits import TRAIN, VAL

logger = logging.getLogger(__name__)

type Reflector = Runnable[Sequence[BaseMessage], Any]

_JSON_INDENT = 2


class PromptText(BaseModel):
    """One revised prompt: its name and its full new text."""

    name: str
    text: str


class Reflection(BaseModel):
    """A revision of every prompt, each as its full new text, and why it changed."""

    # A list, not a name-to-text map: strict structured output refuses open maps.
    prompts: list[PromptText]
    rationale: str


@dataclass
class _Lineage:
    """The hill-climb so far: the best candidate, its runs, and every record."""

    run: Run
    baseline: Candidate
    baseline_val: Evaluation
    baseline_redteam: Summary | None
    best: Candidate
    best_val: Evaluation
    previous: Summary
    best_train: Evaluation | None = None
    misses: int = 0
    rejected: list[str] = field(default_factory=list)
    records: list[CandidateRecord] = field(default_factory=list)

    @classmethod
    async def start(cls, run: Run, baseline: Candidate) -> Self:
        """Evaluate the baseline on val and red-team it; it starts as the best."""
        val = await run.golden(baseline, VAL)
        redteam = await run.redteam(baseline)
        lineage = cls(
            run=run,
            baseline=baseline,
            baseline_val=val,
            baseline_redteam=redteam,
            best=baseline,
            best_val=val,
            previous=val.summary,
        )
        lineage._add(
            CandidateRecord(
                id=baseline.id,
                parent=None,
                verdict=settings.verdicts.baseline,
                reason=_joined(floor_failures(val, run.section.floors)),
                cost=baseline.cost,
                val=val.summary,
                judge_coverage=val.judge_coverage,
                redteam=redteam,
                agent_calls=run.agent_calls,
            )
        )
        run.write_prompts(run.run_dir / CANDIDATES_DIR / baseline.id, baseline)
        return lineage

    async def climb(self, reflect: Reflector) -> str:
        """Propose and judge candidates until a stop condition; return which."""
        budget, stops = self.run.section.budget, settings.stops
        for number in range(1, budget.iterations + 1):
            cap = budget.max_agent_calls
            if cap is not None and self.run.agent_calls >= cap:
                return stops.max_agent_calls
            failures = await self._failures(number)
            if not failures:
                return stops.no_failures
            await self._propose(number, reflect, failures)
            if self.misses >= budget.patience:
                return stops.patience
        return stops.iterations

    async def _failures(self, number: int) -> list[ExperimentRow]:
        """Return up to k of the best's failed train rows, a seeded sample per round."""
        if self.best_train is None:
            self.best_train = await self.run.golden(self.best, TRAIN)
        failed = self.best_train.failed
        section = self.run.section
        count = min(section.budget.failures_per_reflection, len(failed))
        # Seeded per round: a retry after a rejection sees other failures, and a
        # rerun with the same seed sees the same ones.
        return random.Random(f"{section.seed}:{number}").sample(failed, count)

    async def _propose(
        self, number: int, reflect: Reflector, failures: Sequence[ExperimentRow]
    ) -> None:
        """Ask for a revision of the best's prompts, then evaluate and judge it."""
        candidate_id = settings.run.candidate_id.format(number=number)
        messages = self._messages(failures)
        try:
            reflection = Reflection.model_validate(await reflect.ainvoke(messages))
        # An outage or unparsable reply is an invalid candidate, which patience bounds.
        except Exception as error:
            name = type(error).__name__
            logger.warning(settings.texts.reflection_failed, candidate_id, name)
            reason = settings.reasons.reflection_failed.format(error=name)
            self._reject(self._invalid(candidate_id, reason, None))
            return
        prompts = _named_prompts(reflection, self.baseline.prompts)
        if prompts is None:
            returned = sorted(prompt.name for prompt in reflection.prompts)
            expected = sorted(self.baseline.prompts)
            reason = settings.reasons.names.format(names=returned, expected=expected)
            self._reject(self._invalid(candidate_id, reason, reflection.rationale))
            return
        candidate = Candidate(candidate_id, prompts, self.best.cost)
        await self._evaluate(candidate, reflection.rationale)

    async def _evaluate(self, candidate: Candidate, rationale: str) -> None:
        """Run the candidate on val, red-team it if it passes, then judge it."""
        val = await self.run.golden(candidate, VAL)
        paired = pair(self.best_val, val)
        reasons = self._gate(val, paired.regressed_cases)
        redteam = None
        if not reasons:
            redteam = await self.run.redteam(candidate)
            limit = self.run.section.max_regression.attack_succeeded
            reasons = redteam_failures(redteam, self.baseline_redteam, limit)
        verdicts = settings.verdicts
        record = CandidateRecord(
            id=candidate.id,
            parent=self.best.id,
            verdict=verdicts.rejected if reasons else verdicts.accepted,
            reason=_joined(reasons),
            rationale=rationale,
            cost=candidate.cost,
            val=val.summary,
            judge_coverage=val.judge_coverage,
            redteam=redteam,
            deltas={
                **deltas(val.summary, self.baseline_val.summary, VS_BASELINE),
                **deltas(val.summary, self.previous, VS_PREVIOUS),
            },
            agent_calls=self.run.agent_calls,
            **asdict(paired),
        )
        self.run.write_prompts(
            self.run.run_dir / CANDIDATES_DIR / candidate.id, candidate
        )
        self.previous = val.summary
        if reasons:
            self._reject(record)
            return
        self._add(record)
        self.best, self.best_val, self.best_train, self.misses = candidate, val, None, 0

    def _gate(self, val: Evaluation, regressed: Sequence[str]) -> list[str]:
        """Return why val is not accepted before red-teaming; [] if it may be."""
        section, reasons = self.run.section, settings.reasons
        failures = floor_failures(val, section.floors)
        value, best = (
            val.summary.get("pass_rate"),
            self.best_val.summary.get("pass_rate"),
        )
        min_delta = section.budget.min_delta
        if value is None or (best is not None and not at_most(best + min_delta, value)):
            message = reasons.min_delta
            failures.append(message.format(value=value, best=best, min_delta=min_delta))
        risky = [case for case in regressed if case in val.high_risk]
        if risky:
            failures.append(reasons.high_risk.format(cases=risky))
        return failures

    def _messages(self, failures: Sequence[ExperimentRow]) -> list[BaseMessage]:
        """Return the reflection's rules, then the prompts, failures and rejections."""
        prompts = settings.prompts
        blocks = prompts.prompt_separator.join(
            prompts.prompt_block.format(name=name, text=text)
            for name, text in self.best.prompts.items()
        )
        excerpts = [_excerpt(row, self.run.graders.pass_keys) for row in failures]
        data = prompts.reflection_prompt.format(
            prompts=blocks,
            failures=_escape_tags(_json(excerpts)),
            rejected=_escape_tags(_json(self.rejected)),
        )
        return [SystemMessage(prompts.reflection_system), HumanMessage(data)]

    def _invalid(
        self, candidate_id: str, reason: str, rationale: str | None
    ) -> CandidateRecord:
        """Return the record of a candidate the reflection could not produce."""
        return CandidateRecord(
            id=candidate_id,
            parent=self.best.id,
            verdict=settings.verdicts.invalid,
            reason=reason,
            rationale=rationale,
            agent_calls=self.run.agent_calls,
        )

    def _reject(self, record: CandidateRecord) -> None:
        """Record a candidate that did not replace the best, and count the miss."""
        self._add(record)
        self.misses += 1
        if record.rationale is not None:
            self.rejected.append(record.rationale)

    def _add(self, record: CandidateRecord) -> None:
        """Append the record and log its verdict."""
        self.records.append(record)
        logger.info(settings.texts.verdict, record.id, record.verdict, record.reason)


async def optimize(
    *,
    config: AgentConfig,
    source: CaseSource,
    agent: Agent,
    model_factory: ModelFactory,
    runner: ExperimentRunner,
    results_dir: Path,
    upload: bool = False,
) -> Path:
    """Hill-climb the agent's [optimize] prompts on val; return the run's folder.

    The baseline is the prompt files, always sent as prompt_overrides. Each
    round, the reflection model revises the best prompts from up to k of their
    failed train cases; the revision is accepted only if its val run clears the
    floors and judge coverage, beats the best's pass_rate by min_delta, breaks no
    high-risk case the best passed, and, when red-teamed, keeps attack_succeeded
    within max_regression of the baseline's with every attack scored. Stops at
    iterations, patience, max_agent_calls or no failures left, then runs the
    baseline and the best on holdout: "adopt" only if the best scores higher.

    Writes results_dir/<run_id>/: report.json, candidates/<id>/<name>.md,
    best/<name>.md and evals/<candidate>-<split>/. The agent's prompt files are
    only read. upload=True uploads each experiment to LangSmith, on one dataset
    per split. Raises AgentConfigError for a bad table and ValueError for bad
    splits before any agent call.
    """
    run = start_run(
        config=config,
        source=source,
        agent=agent,
        model_factory=model_factory,
        runner=runner,
        results_dir=results_dir,
        upload=upload,
        mode=settings.keys.optimize,
    )
    model = model_factory(run.section.reflection_model)
    reflect: Reflector = model.with_structured_output(Reflection)
    prompts = read_prompts(config, run.section.prompts)
    baseline = Candidate(settings.keys.baseline, prompts, run.section.cost)
    lineage = await _Lineage.start(run, baseline)
    stop = await lineage.climb(reflect)
    logger.info(settings.texts.stopped, stop)
    base, best, holdout = await run.holdout(baseline, lineage.best)
    verdict, reason = _verdict(base, best)
    logger.info(settings.texts.final, lineage.best.id, verdict)
    run.write_prompts(run.run_dir / BEST_DIR, lineage.best)
    return run.write_report(
        {
            **run.header(),
            "candidates": [asdict(record) for record in lineage.records],
            "best": lineage.best.id,
            "stop_reason": stop,
            "holdout": holdout,
            "verdict": verdict,
            "reason": reason,
            "agent_calls": run.agent_calls,
        }
    )


def _verdict(baseline: Evaluation, best: Evaluation) -> tuple[str, str | None]:
    """Return "adopt" if best beats the baseline on holdout, else why not."""
    verdicts, reasons = settings.verdicts, settings.reasons
    if best is baseline:
        return verdicts.do_not_adopt, reasons.is_baseline
    value, base = best.summary.get("pass_rate"), baseline.summary.get("pass_rate")
    if value is None or base is None or value <= base:
        message = reasons.not_above_baseline
        return verdicts.do_not_adopt, message.format(value=value, baseline=base)
    return verdicts.adopt, None


def _named_prompts(
    reflection: Reflection, names: Mapping[str, str]
) -> dict[str, str] | None:
    """Return the revision's prompts in names' order; None unless exactly names."""
    returned = [prompt.name for prompt in reflection.prompts]
    if sorted(returned) != sorted(names):
        return None
    texts = {prompt.name: prompt.text for prompt in reflection.prompts}
    return {name: texts[name] for name in names}


def _excerpt(row: ExperimentRow, pass_keys: Sequence[str]) -> dict[str, Any]:
    """Return a failed row's last user message, reply and failing graders' comments."""
    chars = settings.run.excerpt_chars
    messages = row.inputs.get("messages") or [{}]
    return {
        "input": str(messages[-1].get("content", ""))[:chars],
        "reply": str(row.outputs.get("text") or row.error or "")[:chars],
        "comments": {
            key: row.comments.get(key)
            for key in pass_keys
            if row.feedback.get(key) not in (None, 1)
        },
    }


def _json(value: object) -> str:
    """Return value as indented JSON, non-ASCII text kept as is."""
    return json.dumps(value, ensure_ascii=False, indent=_JSON_INDENT)


def _joined(reasons: Sequence[str]) -> str | None:
    """Return the reasons as one string, or None if there are none."""
    return settings.reasons.separator.join(reasons) or None


__all__ = ["PromptText", "Reflection", "optimize"]
