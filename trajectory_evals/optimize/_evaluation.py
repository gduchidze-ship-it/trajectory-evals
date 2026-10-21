"""Candidates' experiments with the graders and floors fixed for a whole run.

The optimizer and the sweep both evaluate through here, so a candidate scores
exactly as an eval run of the same prompts would, and both write the same files.
"""

import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.artifacts import default_run_id, to_json, write_run
from trajectory_evals.core.protocols import Agent, CaseSource
from trajectory_evals.core.runner import (
    ExperimentConfig,
    ExperimentReport,
    ExperimentRow,
    ExperimentRunner,
    Target,
)
from trajectory_evals.core.summary import passed, summarize
from trajectory_evals.golden.graders import Graders, ModelFactory, build_graders
from trajectory_evals.golden.settings import settings as golden_settings
from trajectory_evals.golden.target import GoldenTarget
from trajectory_evals.optimize.settings import (
    Floors,
    OptimizeSection,
    optimize_table,
    settings,
    table_error,
)
from trajectory_evals.optimize.splits import HOLDOUT, Splits, split_cases
from trajectory_evals.simulation import (
    RedteamSection,
    SimulationTarget,
    agent_table,
    create_async_attack_succeeded_evaluator,
    create_async_turns_to_break_evaluator,
    load_personas,
    redteam_cases,
)
from trajectory_evals.simulation.settings import settings as simulation_settings

logger = logging.getLogger(__name__)

type Summary = dict[str, float | None]

REPORT_FILE = "report.json"
CANDIDATES_DIR = "candidates"
BEST_DIR = "best"
EVALS_DIR = "evals"
VS_BASELINE = "delta_vs_baseline"
VS_PREVIOUS = "delta_vs_previous"
_REPORT_INDENT = 2

_REDTEAM_GRADERS = Graders(
    evaluators=(
        create_async_attack_succeeded_evaluator(),
        create_async_turns_to_break_evaluator(),
    )
)


@dataclass(frozen=True)
class Candidate:
    """Prompt texts by prompt name, under an id, at a declared cost."""

    id: str
    prompts: Mapping[str, str]
    cost: float


@dataclass(frozen=True)
class Evaluation:
    """One candidate's golden run on one split.

    passes says, per case, whether every counted run of it passed. failed holds
    one failed row per case, in case id order. judge_coverage is each judge's
    share of scored rows among those it was due to score: a row it failed on
    counts against it, a row it does not apply to does not; None if none was due.
    """

    summary: Summary
    passes: Mapping[str, bool]
    high_risk: frozenset[str]
    failed: tuple[ExperimentRow, ...]
    judge_coverage: Mapping[str, float | None]


@dataclass(frozen=True)
class Paired:
    """Cases a candidate fixed and broke against its parent, and the sign test.

    sign_test_p is the exact one-sided p-value that the candidate is no better,
    over the discordant cases only.
    """

    fixed_cases: tuple[str, ...]
    regressed_cases: tuple[str, ...]
    sign_test_p: float


@dataclass(frozen=True)
class CandidateRecord:
    """One candidate's entry in report.json.

    verdict is a [verdicts] value and reason joins every [reasons] message that
    applies, None when none does. val and redteam are summarize() outputs, None
    when that run did not happen. fixed_cases and regressed_cases compare val
    with the parent's; deltas hold <key>_delta_vs_baseline and
    <key>_delta_vs_previous, against the candidate evaluated just before.
    agent_calls is the run's total so far.
    """

    id: str
    parent: str | None
    verdict: str
    reason: str | None = None
    rationale: str | None = None
    cost: float | None = None
    val: Summary | None = None
    judge_coverage: Mapping[str, float | None] = field(default_factory=dict)
    redteam: Summary | None = None
    fixed_cases: tuple[str, ...] = ()
    regressed_cases: tuple[str, ...] = ()
    sign_test_p: float | None = None
    deltas: Mapping[str, float] = field(default_factory=dict)
    agent_calls: int = 0


@dataclass
class Run:
    """One optimize or sweep run: its folder, splits, graders and agent calls so far.

    The graders, and so every judge, are built once, so candidates are compared
    under the same judge. attacks is the red-team source, None when off.
    """

    config: AgentConfig
    section: OptimizeSection
    agent: Agent
    model_factory: ModelFactory
    runner: ExperimentRunner
    upload: bool
    mode: str
    run_dir: Path
    splits: Splits
    graders: Graders
    judge_keys: tuple[str, ...]
    attacks: CaseSource | None
    agent_calls: int = 0

    async def golden(self, candidate: Candidate, split: str) -> Evaluation:
        """Run and grade the candidate's prompts on a split's golden cases."""
        target = GoldenTarget(self.agent, prompt_overrides=candidate.prompts)
        source: CaseSource = getattr(self.splits, split)
        report = await self._run(candidate, split, source, target, self.graders)
        # Each row is one send to the agent; rows already include repetitions.
        self.agent_calls += len(report.rows)
        pass_keys = self.graders.pass_keys
        summary = summarize(
            report.rows, pass_keys=pass_keys, segment_by=(settings.keys.risk,)
        )
        # A [graders] factory's summary scores, such as a confusion matrix, join
        # in; summarize's own keys win, as in the CLI, since the floors read them.
        summary |= {k: v for k, v in report.summary.items() if k not in summary}
        self._write(report, candidate, split, summary)
        logger.info(settings.texts.evaluated, candidate.id, split, summary["pass_rate"])
        return _evaluation(report.rows, summary, pass_keys, self.judge_keys)

    async def redteam(self, candidate: Candidate) -> Summary | None:
        """Run the red-team attacks on the candidate's prompts; None when off."""
        if self.attacks is None:
            return None
        target = SimulationTarget(
            self.agent,
            config=self.config,
            model_factory=self.model_factory,
            prompt_overrides=candidate.prompts,
        )
        split = settings.keys.redteam
        report = await self._run(
            candidate, split, self.attacks, target, _REDTEAM_GRADERS
        )
        # A red-team row is a conversation: each turn it reports is one send. An
        # errored row reports none, so its sends before the error go uncounted.
        self.agent_calls += sum(row.outputs.get("turns") or 0 for row in report.rows)
        summary = summarize(report.rows)
        self._write(report, candidate, split, summary)
        return summary

    async def holdout(
        self, baseline: Candidate, chosen: Candidate
    ) -> tuple[Evaluation, Evaluation, dict[str, Any]]:
        """Run the baseline and the chosen candidate, once if the same, on holdout.

        Returns both evaluations and their comparison for report.json.
        """
        base = await self.golden(baseline, HOLDOUT)
        best = base if chosen.id == baseline.id else await self.golden(chosen, HOLDOUT)
        comparison = {
            "candidate": chosen.id,
            "baseline": base.summary,
            "best": best.summary,
            "deltas": deltas(best.summary, base.summary, VS_BASELINE),
            **asdict(pair(base, best)),
        }
        return base, best, comparison

    def header(self) -> dict[str, Any]:
        """Return the report.json keys every run starts with."""
        return {
            "run_id": self.run_dir.name,
            "agent": self.config.name,
            "mode": self.mode,
            "upload": self.upload,
            "optimize": self.section.model_dump(),
            "splits": {**self.splits.sizes(), "labelled": self.splits.labelled},
        }

    def write_prompts(self, folder: Path, candidate: Candidate) -> None:
        """Write each of the candidate's prompts to <name>.md in a new folder."""
        folder.mkdir(parents=True)
        for name, text in candidate.prompts.items():
            path = folder / settings.run.prompt_file.format(name=name)
            path.write_text(text, encoding="utf-8")

    def write_report(self, report: Mapping[str, Any]) -> Path:
        """Write report.json into the run folder and return the folder."""
        text = to_json(report, _REPORT_INDENT)
        (self.run_dir / REPORT_FILE).write_text(text, encoding="utf-8")
        return self.run_dir

    async def _run(
        self,
        candidate: Candidate,
        split: str,
        source: CaseSource,
        target: Target,
        graders: Graders,
    ) -> ExperimentReport:
        """Run one experiment of the candidate on source, named for its split."""
        run = settings.run
        names = {
            "agent": self.config.name,
            "mode": self.mode,
            "seed": self.section.seed,
            "candidate": candidate.id,
            "split": split,
        }
        return await self.runner.run(
            source=source,
            dataset_name=run.dataset_name.format(**names),
            target=target,
            evaluators=graders.evaluators,
            summary_evaluators=graders.summary_evaluators,
            config=ExperimentConfig(
                experiment_prefix=run.experiment_prefix.format(**names),
                max_concurrency=run.max_concurrency,
                num_repetitions=self.section.budget.repetitions,
                upload_results=self.upload,
            ),
            metadata=names,
        )

    def _write(
        self,
        report: ExperimentReport,
        candidate: Candidate,
        split: str,
        summary: Summary,
    ) -> None:
        """Write the experiment's rows and summary to evals/<candidate>-<split>/."""
        meta = {
            "agent": self.config.name,
            "mode": self.mode,
            "candidate": candidate.id,
            "split": split,
            "experiment": report.experiment_name,
            "dataset": report.dataset_name,
            "url": report.url,
        }
        write_run(
            report,
            results_dir=self.run_dir / EVALS_DIR,
            summary=summary,
            meta=meta,
            run_id=settings.run.eval_run_id.format(candidate=candidate.id, split=split),
        )


def start_run(
    *,
    config: AgentConfig,
    source: CaseSource,
    agent: Agent,
    model_factory: ModelFactory,
    runner: ExperimentRunner,
    results_dir: Path,
    upload: bool,
    mode: str,
) -> Run:
    """Validate the agent's tables, split the cases and build the graders.

    The criteria judge is built only when some case has expected.criteria. The
    run folder is made by the first experiment written into it. Raises
    AgentConfigError for a bad table, including red-teaming switched on without
    a [redteam] table or its policy, and ValueError for bad splits.
    """
    section = optimize_table(config)
    splits = split_cases(source, section.split, seed=section.seed)
    criteria = any(case["outputs"].get("criteria") for case in source.examples())
    graders = build_graders(config, model_factory=model_factory, criteria=criteria)
    criteria_keys = (golden_settings.keys.criteria,) if criteria else ()
    name = settings.run.run_name.format(agent=config.name, mode=mode)
    return Run(
        config=config,
        section=section,
        agent=agent,
        model_factory=model_factory,
        runner=runner,
        upload=upload,
        mode=mode,
        run_dir=results_dir / default_run_id(name),
        splits=splits,
        graders=graders,
        judge_keys=(*criteria_keys, *config.section("judges")),
        attacks=attack_source(config, section),
    )


def attack_source(config: AgentConfig, section: OptimizeSection) -> CaseSource | None:
    """Return the red-team cases, None when [optimize].redteam is off.

    Raises AgentConfigError, before any agent call, when red-teaming is on but
    the agent has no [redteam] table, or the table no readable policy: the gate
    would otherwise be skipped, or fail on every attack.
    """
    if not section.redteam:
        return None
    texts = settings.texts
    if not config.section("redteam"):
        raise table_error(config, "optimize", texts.no_redteam)
    redteam = agent_table(config, "redteam", RedteamSection)
    if redteam.policy is None:
        raise table_error(config, "redteam", texts.no_policy)
    config.read_text(redteam.policy)
    personas = load_personas(config)
    return redteam_cases(personas, names=redteam.personas, attempts=redteam.attempts)


def _evaluation(
    rows: Sequence[ExperimentRow],
    summary: Summary,
    pass_keys: Sequence[str],
    judge_keys: Sequence[str],
) -> Evaluation:
    """Return the run's summary, per-case passes, high-risk cases and failures."""
    runs: dict[str, list[bool]] = defaultdict(list)
    failed: dict[str, ExperimentRow] = {}
    for row in rows:
        outcome = passed(row, pass_keys)
        if outcome is not None:
            runs[row.case_id].append(outcome)
        if outcome is False:
            failed.setdefault(row.case_id, row)
    keys = settings.keys
    return Evaluation(
        summary=summary,
        passes={case: all(results) for case, results in runs.items()},
        high_risk=frozenset(
            row.case_id for row in rows if row.metadata.get(keys.risk) == keys.high_risk
        ),
        failed=tuple(failed[case] for case in sorted(failed)),
        judge_coverage={key: _judge_coverage(rows, key) for key in judge_keys},
    )


def _judge_coverage(rows: Sequence[ExperimentRow], key: str) -> float | None:
    """Return the share of rows due the judge's score that got one, None if none."""
    scored = sum(row.feedback.get(key) is not None for row in rows)
    failed = sum(_judge_failed(row, key) for row in rows)
    return scored / (scored + failed) if scored + failed else None


def _judge_failed(row: ExperimentRow, key: str) -> bool:
    """Return whether the judge failed on the row, rather than not applying to it."""
    comment = row.comments.get(key)
    return key in row.feedback_errors or comment == golden_settings.comments.judge_error


def floor_failures(evaluation: Evaluation, floors: Floors) -> list[str]:
    """Return why the evaluation misses each floor it misses; [] if it clears all."""
    summary, reasons = evaluation.summary, settings.reasons
    pass_rate, error_rate = summary.get("pass_rate"), summary.get("error_rate")
    latency = summary.get("latency_p95_ms")
    failures = []
    if pass_rate is None or pass_rate < floors.pass_rate:
        message = reasons.pass_rate_floor
        failures.append(message.format(value=pass_rate, floor=floors.pass_rate))
    if error_rate is None or error_rate > floors.max_error_rate:
        message = reasons.error_rate_floor
        failures.append(message.format(value=error_rate, floor=floors.max_error_rate))
    ceiling = floors.latency_p95_ms
    if ceiling is not None and (latency is None or latency > ceiling):
        failures.append(reasons.latency_floor.format(value=latency, floor=ceiling))
    minimum = floors.min_judge_coverage
    failures.extend(
        reasons.judge_coverage.format(key=key, value=coverage, floor=minimum)
        for key, coverage in evaluation.judge_coverage.items()
        if coverage is not None and coverage < minimum
    )
    return failures


def redteam_failures(
    summary: Summary | None, baseline: Summary | None, limit: float
) -> list[str]:
    """Return why a red-team run fails against the baseline's; [] if none ran.

    Every attack of both runs must be scored, so a judge or agent outage can
    neither pass as a defended attack nor inflate the baseline's rate, which
    would loosen the gate; attack_succeeded may then exceed the baseline's by at
    most limit.
    """
    if summary is None:
        return []
    key, reasons = simulation_settings.keys.attack_succeeded, settings.reasons
    coverage = summary.get(f"{key}_coverage")
    if coverage != 1:
        return [reasons.attack_coverage.format(value=coverage)]
    base_coverage = (baseline or {}).get(f"{key}_coverage")
    if base_coverage != 1:
        return [reasons.baseline_attack_coverage.format(value=base_coverage)]
    value, base = summary.get(key), (baseline or {}).get(key)
    if value is None or base is None or not at_most(value, base + limit):
        message = reasons.attack_succeeded
        return [message.format(value=value, baseline=base, limit=limit)]
    return []


def high_risk_failures(parent: Evaluation, child: Evaluation) -> list[str]:
    """Return why child breaks a high-risk case parent passed; [] if it breaks none.

    A case child left unscored counts as broken, so a judge outage cannot hide
    a regression.
    """
    broken = sorted(
        case
        for case in child.high_risk
        if parent.passes.get(case) and not child.passes.get(case)
    )
    return [settings.reasons.high_risk.format(cases=broken)] if broken else []


def at_most(value: float, bound: float) -> bool:
    """Return whether value is at most bound, allowing float rounding."""
    return value <= bound or math.isclose(value, bound)


def pair(parent: Evaluation, child: Evaluation) -> Paired:
    """Return the cases child fixed and broke against parent, and the sign test."""
    shared = sorted(parent.passes.keys() & child.passes.keys())
    fixed = tuple(c for c in shared if child.passes[c] and not parent.passes[c])
    broken = tuple(c for c in shared if parent.passes[c] and not child.passes[c])
    return Paired(fixed, broken, sign_test(len(fixed), len(broken)))


def sign_test(wins: int, losses: int) -> float:
    """Return the exact one-sided sign-test p-value of wins against losses.

    It is the chance of at least wins successes in wins + losses fair coin
    flips; with no discordant cases it is 1.
    """
    trials = wins + losses
    tail = sum(math.comb(trials, count) for count in range(wins, trials + 1))
    return tail / (1 << trials)  # 2**trials as an int, exact for any count


def deltas(summary: Summary, other: Summary, suffix: str) -> dict[str, float]:
    """Return summary minus other for each key both score, as <key>_<suffix>."""
    return {
        f"{key}_{suffix}": value - other_value
        for key, value in summary.items()
        if value is not None and (other_value := other.get(key)) is not None
    }


__all__ = [
    "BEST_DIR",
    "CANDIDATES_DIR",
    "EVALS_DIR",
    "REPORT_FILE",
    "VS_BASELINE",
    "VS_PREVIOUS",
    "Candidate",
    "CandidateRecord",
    "Evaluation",
    "Paired",
    "Run",
    "Summary",
    "at_most",
    "attack_source",
    "deltas",
    "floor_failures",
    "high_risk_failures",
    "pair",
    "redteam_failures",
    "sign_test",
    "start_run",
]
