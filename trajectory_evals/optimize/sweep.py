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
    Candidate,
    CandidateRecord,
    Evaluation,
    Run,
    Summary,
    deltas,
    floor_failures,
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
    the baseline's cost. Each runs on val, and is red-teamed against the baseline
    when it clears the floors; the lowest cost wins, then the higher pass_rate,
    then the lower latency_p50_ms. The winner and the baseline then run on
    holdout: "adopt" only if the winner is not the baseline and clears the
    floors there too.

    Writes results_dir/<run_id>/report.json and evals/<candidate>-<split>/.
    Raises AgentConfigError for a bad table, ValueError for bad splits and
    OSError for an include file it cannot read, before any agent call.
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
    candidates = _candidates(config, run.section, include)
    entries = await _evaluate(run, candidates)
    passing = [entry for entry in entries if not entry.reasons]
    winner = min(passing, key=_rank, default=None)
    report: dict[str, Any] = {
        **run.header(),
        "variants": [_record(entry, entries[0]) for entry in entries],
        "winner": None if winner is None else winner.candidate.id,
    }
    report |= await _confirm(run, candidates[0], winner)
    return run.write_report({**report, "agent_calls": run.agent_calls})


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


async def _evaluate(run: Run, candidates: list[Candidate]) -> list[_Entry]:
    """Run each candidate on val, and red-team the baseline and those that pass."""
    floors = run.section.floors
    limit = run.section.max_regression.attack_succeeded
    entries: list[_Entry] = []
    for candidate in candidates:
        val = await run.golden(candidate, VAL)
        reasons = floor_failures(val, floors)
        # The baseline comes first and is always red-teamed: the others compare
        # their attack rate with its.
        is_baseline = not entries
        redteam = None
        if is_baseline or not reasons:
            redteam = await run.redteam(candidate)
        baseline_redteam = redteam if is_baseline else entries[0].redteam
        if not reasons:
            reasons = redteam_failures(redteam, baseline_redteam, limit)
        entries.append(_Entry(candidate, val, redteam, tuple(reasons)))
    return entries


def _rank(entry: _Entry) -> tuple[float, float, float]:
    """Return the sort key: lowest cost, then highest pass_rate, then lowest p50."""
    summary = entry.val.summary
    latency = summary.get("latency_p50_ms")
    return (
        entry.candidate.cost,
        -(summary.get("pass_rate") or 0.0),
        math.inf if latency is None else latency,
    )


def _record(entry: _Entry, baseline: _Entry) -> dict[str, Any]:
    """Return the entry's row of the variant table, compared with the baseline."""
    verdicts = settings.verdicts
    paired = pair(baseline.val, entry.val)
    record = CandidateRecord(
        id=entry.candidate.id,
        parent=None if entry is baseline else baseline.candidate.id,
        verdict=verdicts.rejected if entry.reasons else verdicts.passed,
        reason=settings.reasons.separator.join(entry.reasons) or None,
        cost=entry.candidate.cost,
        val=entry.val.summary,
        judge_coverage=entry.val.judge_coverage,
        redteam=entry.redteam,
        deltas=deltas(entry.val.summary, baseline.val.summary, VS_BASELINE),
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
