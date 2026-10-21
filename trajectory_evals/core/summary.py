"""Pass rates, feedback means, errors and latency over an experiment's rows."""

import math
import statistics
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence

from trajectory_evals.core.runner import ExperimentRow
from trajectory_evals.core.settings import settings

_WILSON_Z = 1.96  # two-sided 95%
_TWENTIETHS = 20  # quantiles cut at 5% steps, so the last cut is the 95th percentile
COVERAGE_SUFFIX = "_coverage"  # <key>_coverage is the share of rows that scored key
SUMMARY_KEYS = frozenset(
    {
        "pass_rate",
        "pass_rate_ci_low",
        "pass_rate_ci_high",
        "pass_all_k",
        "pass_any_k",
        "repetitions",
        "error_rate",
        "latency_p50_ms",
        "latency_p95_ms",
        "rows",
    }
)


def summarize(
    rows: Sequence[ExperimentRow],
    *,
    pass_keys: Collection[str] = (),
    segment_by: Sequence[str] = (),
) -> dict[str, float | None]:
    """Return the experiment's scores by key; a score with no data is None.

    <key> is the mean of a feedback key's scored values (a bool counts 0 or 1) and
    <key>_coverage the share of rows that scored it, in key order; a feedback key
    that is also a summary key raises ValueError. A row counts toward pass_rate
    when it scored at least one pass key, and passes only if each one it scored
    is 1 (True), so a judge pass never overrides a deterministic fail. With pass
    keys, a row whose target raised or any of whose graders raised counts as a
    fail; error_rate still reports errors on its own. pass_rate_ci_low and
    pass_rate_ci_high are pass_rate's Wilson 95% interval with n the number of
    distinct counted cases, so repetitions do not narrow it. When a case_id
    repeats, pass_all_k and pass_any_k are the shares of cases whose counted runs
    all, or any, passed, and repetitions the fewest runs of any case.
    error_rate, latency_p50_ms, latency_p95_ms and rows follow.
    pass_rate[<segment>=<value>] reads value from the row's metadata, else, when
    that is missing or None, from its outputs, else from its inputs, so a row
    whose target raised keeps its input segment; rows with none are left out.
    """
    passes = [passed(row, pass_keys) for row in rows]
    pass_rate = _mean(passes)
    counted = zip(rows, passes, strict=True)
    cases = {row.case_id for row, outcome in counted if outcome is not None}
    low, high = _wilson_interval(pass_rate, len(cases))
    latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
    return {
        **_feedback_means(rows),
        "pass_rate": pass_rate,
        "pass_rate_ci_low": low,
        "pass_rate_ci_high": high,
        **_pass_at_k(rows, passes),
        "error_rate": _mean([row.error is not None for row in rows]),
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": _p95(latencies),
        "rows": len(rows),
        **_segment_pass_rates(rows, passes, segment_by),
    }


def _mean(values: Iterable[float | bool | None]) -> float | None:
    """Return the mean of the values that are not None, or None if none are."""
    scored = [float(value) for value in values if value is not None]
    return statistics.fmean(scored) if scored else None


def passed(row: ExperimentRow, pass_keys: Collection[str]) -> bool | None:
    """Return whether the row passed every pass key it scored, or None if none."""
    if not pass_keys:
        return None
    # aevaluate files a crashed grader under its function's source name, which for
    # a wrapped grader is not the pass key it scores, so any crash fails the row.
    if row.error is not None or row.feedback_errors:
        return False
    scores = [row.feedback.get(key) for key in pass_keys]
    scored = [score for score in scores if score is not None]
    return all(score == 1 for score in scored) if scored else None


def _feedback_means(rows: Sequence[ExperimentRow]) -> dict[str, float | None]:
    """Return each feedback key's mean and coverage, keys sorted."""
    keys = sorted({key for row in rows for key in row.feedback})
    coverage_keys = {key + COVERAGE_SUFFIX for key in keys}
    collisions = set(keys) & (SUMMARY_KEYS | coverage_keys)
    if collisions:
        message = settings.texts.summary_key_collision
        raise ValueError(message.format(keys=sorted(collisions)))
    means: dict[str, float | None] = {}
    for key in keys:
        scores = [row.feedback.get(key) for row in rows]
        means[key] = _mean(scores)
        means[key + COVERAGE_SUFFIX] = _mean([score is not None for score in scores])
    return means


def _wilson_interval(
    rate: float | None, counted: int
) -> tuple[float | None, float | None]:
    """Return the Wilson score 95% interval of rate over counted trials, or Nones."""
    if rate is None or not counted:
        return None, None
    z_squared = _WILSON_Z**2
    scale = 1 + z_squared / counted
    centre = (rate + z_squared / (2 * counted)) / scale
    spread = rate * (1 - rate) / counted + z_squared / (4 * counted**2)
    half_width = _WILSON_Z * math.sqrt(spread) / scale
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _pass_at_k(
    rows: Sequence[ExperimentRow], passes: Sequence[bool | None]
) -> dict[str, float | None]:
    """Return pass_all_k, pass_any_k and repetitions, or nothing if no case repeats."""
    by_case: dict[str, list[bool | None]] = defaultdict(list)
    for row, outcome in zip(rows, passes, strict=True):
        by_case[row.case_id].append(outcome)
    if all(len(case_passes) == 1 for case_passes in by_case.values()):
        return {}
    counted = [
        [outcome for outcome in case_passes if outcome is not None]
        for case_passes in by_case.values()
    ]
    counted = [case_passes for case_passes in counted if case_passes]
    return {
        "pass_all_k": _mean([all(case_passes) for case_passes in counted]),
        "pass_any_k": _mean([any(case_passes) for case_passes in counted]),
        "repetitions": min(len(case_passes) for case_passes in by_case.values()),
    }


def _p95(latencies: Sequence[float]) -> float | None:
    """Return the 95th percentile of latencies, interpolated, or None if empty."""
    if not latencies:
        return None
    return statistics.quantiles(latencies, n=_TWENTIETHS, method="inclusive")[-1]


def _segment_pass_rates(
    rows: Sequence[ExperimentRow],
    passes: Sequence[bool | None],
    segment_by: Sequence[str],
) -> dict[str, float | None]:
    """Return pass_rate[<segment>=<value>] per segment value, sorted by key."""
    groups: dict[str, list[bool | None]] = defaultdict(list)
    for segment in segment_by:
        for row, outcome in zip(rows, passes, strict=True):
            value = _segment_value(row, segment)
            if value is not None:
                groups[f"pass_rate[{segment}={value}]"].append(outcome)
    return {key: _mean(groups[key]) for key in sorted(groups)}


def _segment_value(row: ExperimentRow, segment: str) -> object:
    """Return the row's segment value from metadata, else outputs, else inputs."""
    for values in (row.metadata, row.outputs, row.inputs):
        value = values.get(segment)
        if value is not None:
            return value
    return None


__all__ = ["COVERAGE_SUFFIX", "SUMMARY_KEYS", "passed", "summarize"]
