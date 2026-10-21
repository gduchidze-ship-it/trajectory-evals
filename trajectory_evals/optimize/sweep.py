"""The variant sweep: the cheapest declared prompt variant that clears every floor."""

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.protocols import Agent, CaseSource
from trajectory_evals.core.runner import ExperimentRunner
from trajectory_evals.golden.graders import ModelFactory
from trajectory_evals.optimize._evaluation import (
    VS_BASELINE,
    VS_PREVIOUS,
    Candidate,
    CandidateRecord,
    Evaluation,
    Run,
    Summary,
    deltas,
    floor_failures,
    high_risk_failures,
    pair,
    redteam_failures,
    start_run,
)
from trajectory_evals.optimize.settings import (
    OptimizeSection,
    read_prompts,
    settings,
    variant_tables,
)
from trajectory_evals.optimize.splits import VAL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Entry:
    """One candidate's val run, red-team run, and why it misses the floors."""

    candidate: Candidate
    val: Evaluation
    redteam: Summary | None
    reasons: tuple[str, ...]


async def sweep(
    *,
    config: AgentConfig,
    source: CaseSource,
    agent: Agent,
    model_factory: ModelFactory,
    runner: ExperimentRunner,
    results_dir: Path,
    upload: bool = False,
    include: Path | None = None,
) -> Path:
    """Pick the cheapest prompt variant that clears the floors; return the folder.

    Candidates are the baseline (the [optimize] prompt files at [optimize].cost),
    each [variants.<name>] (its files over the baseline's, at its cost) and, with
    include, the <name>.md files of a folder such as an optimize run's best/ at
    the baseline's cost. Each runs on val; one that clears the floors and breaks
    no high-risk case the baseline passed is red-teamed against the baseline.
    The lowest cost wins, then the higher pass_rate, then the lower
    latency_p50_ms. The winner and the baseline then run on holdout: "adopt"
    only if the winner is not the baseline and clears the floors there too.

    Writes results_dir/<run_id>/report.json, rewritten after every candidate
    and completed at the end, and evals/<candidate>-<split>/. Raises
    AgentConfigError for a bad table, or red-teaming on without a [redteam]
    table and policy, ValueError for bad splits and OSError for an include file
    it cannot read, all before any agent call.
    """
    run = start_run(
        config=config,
        source=source,
        agent=agent,
        model_factory=model_factory,
        runner=runner,
        results_dir=results_dir,
        upload=upload,
        mode=settings.keys.sweep,
    )
    entries: list[_Entry] = []
    for candidate in _candidates(config, run.section, include):
        entries.append(await _evaluate(run, candidate, entries))
        # Written after every candidate, so a crash later keeps these rows.
        run.write_report(_report(run, entries))
    passing = [entry for entry in entries if not entry.reasons]
    winner = min(passing, key=_rank, default=None)
    final = await _confirm(run, entries[0].candidate, winner)
    winner_id = None if winner is None else winner.candidate.id
    return run.write_report(_report(run, entries, winner=winner_id, **final))


def _candidates(
    config: AgentConfig, section: OptimizeSection, include: Path | None
) -> list[Candidate]:
    """Return the baseline, each variant, then the included folder's prompts."""
    keys = settings.keys
    prompts = read_prompts(config, section.prompts)
    candidates = [Candidate(keys.baseline, prompts, section.cost)]
    candidates.extend(
        Candidate(
            name,
            {**prompts, **read_prompts(config, variant.prompt_overrides)},
            variant.cost,
        )
        for name, variant in variant_tables(config, section).items()
    )
    if include is not None:
        prompt_file = settings.run.prompt_file
        included = {
            name: (include / prompt_file.format(name=name)).read_text(encoding="utf-8")
            for name in prompts
        }
        candidates.append(Candidate(keys.include, included, section.cost))
    return candidates


async def _evaluate(run: Run, candidate: Candidate, earlier: list[_Entry]) -> _Entry:
    """Run the candidate on val, and red-team it if it is the baseline or passes.

    The baseline comes first and is always red-teamed: the others compare their
    high-risk cases and attack rate with its.
    """
    val = await run.golden(candidate, VAL)
    baseline = earlier[0] if earlier else None
    reasons = floor_failures(val, run.section.floors)
    if baseline is not None:
        reasons += high_risk_failures(baseline.val, val)
    redteam = None
    if baseline is None or not reasons:
        redteam = await run.redteam(candidate)
    if not reasons:
        base_redteam = redteam if baseline is None else baseline.redteam
        limit = run.section.max_regression.attack_succeeded
        reasons = redteam_failures(redteam, base_redteam, limit)
    return _Entry(candidate, val, redteam, tuple(reasons))


def _rank(entry: _Entry) -> tuple[float, float, float]:
    """Return the sort key: lowest cost, then highest pass_rate, then lowest p50."""
    summary = entry.val.summary
    latency = summary.get("latency_p50_ms")
    return (
        entry.candidate.cost,
        -(summary.get("pass_rate") or 0.0),
        math.inf if latency is None else latency,
    )


def _report(run: Run, entries: list[_Entry], **final: Any) -> dict[str, Any]:
    """Return report.json so far; final sets the winner, holdout and verdict."""
    previous = [None, *entries[:-1]]
    return {
        **run.header(),
        "variants": [
            _record(entry, entries[0], before)
            for entry, before in zip(entries, previous, strict=True)
        ],
        "winner": None,
        "holdout": None,
        "verdict": None,
        "reason": None,
        **final,
        "agent_calls": run.agent_calls,
    }


def _record(entry: _Entry, baseline: _Entry, previous: _Entry | None) -> dict[str, Any]:
    """Return the entry's row of the variant table, compared with the baseline.

    deltas also compare it with the previous entry in variant order, if any.
    """
    verdicts = settings.verdicts
    paired = pair(baseline.val, entry.val)
    before = {} if previous is None else previous.val.summary
    record = CandidateRecord(
        id=entry.candidate.id,
        parent=None if entry is baseline else baseline.candidate.id,
        verdict=verdicts.rejected if entry.reasons else verdicts.passed,
        reason=settings.reasons.separator.join(entry.reasons) or None,
        cost=entry.candidate.cost,
        val=entry.val.summary,
        judge_coverage=entry.val.judge_coverage,
        redteam=entry.redteam,
        deltas={
            **deltas(entry.val.summary, baseline.val.summary, VS_BASELINE),
            **deltas(entry.val.summary, before, VS_PREVIOUS),
        },
        **asdict(paired),
    )
    return asdict(record)


async def _confirm(
    run: Run, baseline: Candidate, winner: _Entry | None
) -> dict[str, Any]:
    """Run the winner and the baseline on holdout; return the holdout and verdict."""
    verdicts, reasons = settings.verdicts, settings.reasons
    if winner is None:
        return {
            "holdout": None,
            "verdict": verdicts.do_not_adopt,
            "reason": reasons.no_winner,
        }
    _, best, holdout = await run.holdout(baseline, winner.candidate)
    failures = floor_failures(best, run.section.floors)
    if winner.candidate.id == baseline.id:
        failures = [reasons.is_baseline]
    verdict = verdicts.do_not_adopt if failures else verdicts.adopt
    logger.info(settings.texts.final, winner.candidate.id, verdict)
    reason = reasons.separator.join(failures) or None
    return {"holdout": holdout, "verdict": verdict, "reason": reason}


__all__ = ["sweep"]
