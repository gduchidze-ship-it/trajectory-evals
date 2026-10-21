"""Tests for how the sweep ranks candidates that clear the floors."""

from trajectory_evals.optimize._evaluation import Candidate, Evaluation
from trajectory_evals.optimize.sweep import _Entry, _rank


def _entry(name: str, cost: float, pass_rate: float, p50: float | None) -> _Entry:
    """Return a passing entry with the given cost, pass_rate and median latency."""
    summary = {"pass_rate": pass_rate, "latency_p50_ms": p50}
    val = Evaluation(summary, {}, frozenset(), (), {})
    return _Entry(Candidate(name, {}, cost), val, None, ())


def test_rank_orders_by_cost_then_pass_rate_then_latency() -> None:
    """Cost decides first, a higher pass_rate breaks a tie, then a lower p50."""
    entries = [
        _entry("pricey", 2.0, 1.0, 1.0),
        _entry("slow", 1.0, 0.8, 90.0),
        _entry("unknown", 1.0, 0.8, None),
        _entry("fast", 1.0, 0.8, 10.0),
        _entry("worse", 1.0, 0.7, 1.0),
    ]

    ranked = sorted(entries, key=_rank)

    assert [entry.candidate.id for entry in ranked] == [
        "fast",
        "slow",
        "unknown",
        "worse",
        "pricey",
    ]
