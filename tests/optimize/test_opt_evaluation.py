"""Tests for the floors, the paired view and the red-team gate candidates share."""

from collections.abc import Mapping

import pytest

from tests.core.core_fakes import make_row
from trajectory_evals.golden.settings import settings as golden_settings
from trajectory_evals.optimize._evaluation import (
    Evaluation,
    Summary,
    _evaluation,
    deltas,
    floor_failures,
    pair,
    redteam_failures,
    sign_test,
)
from trajectory_evals.optimize.settings import Floors

FLOORS = Floors(pass_rate=0.5, max_error_rate=0.1, latency_p95_ms=100)
CLEAR: Summary = {"pass_rate": 0.5, "error_rate": 0.1, "latency_p95_ms": 100.0}


def _evaluated(
    summary: Summary,
    passes: Mapping[str, bool] | None = None,
    coverage: Mapping[str, float | None] | None = None,
) -> Evaluation:
    """Return an evaluation with only the fields a test sets."""
    return Evaluation(summary, passes or {}, frozenset(), (), coverage or {})


@pytest.mark.parametrize(
    ("wins", "losses", "p_value"),
    [(5, 0, 0.03125), (0, 0, 1.0), (3, 1, 0.3125), (0, 4, 1.0), (1, 1, 0.75)],
)
def test_sign_test_is_the_exact_one_sided_tail(
    wins: int, losses: int, p_value: float
) -> None:
    """The p-value is P(at least wins of wins + losses fair flips)."""
    assert sign_test(wins, losses) == p_value


def test_pair_lists_fixed_and_regressed_cases_against_the_parent() -> None:
    """Only cases both runs counted are compared, and only discordant ones test."""
    parent = _evaluated({}, {"a": False, "b": True, "c": True, "d": False})
    child = _evaluated({}, {"a": True, "b": False, "c": True, "e": True})

    paired = pair(parent, child)

    assert (paired.fixed_cases, paired.regressed_cases) == (("a",), ("b",))
    assert paired.sign_test_p == 0.75


def test_a_summary_on_every_floor_clears_them() -> None:
    """Values exactly at each floor pass."""
    assert (
        floor_failures(_evaluated(CLEAR, coverage={"j": 0.9, "k": None}), FLOORS) == []
    )


FLOOR_MISSES = {
    "pass_rate": ({"pass_rate": 0.4}, "pass_rate 0.4 is below the floor 0.5"),
    "no_pass_rate": ({"pass_rate": None}, "pass_rate None is below the floor 0.5"),
    "error_rate": ({"error_rate": 0.2}, "error_rate 0.2 is above the floor 0.1"),
    "latency": ({"latency_p95_ms": 101.0}, "latency_p95_ms 101.0 is above"),
}


@pytest.mark.parametrize(
    ("override", "reason"), FLOOR_MISSES.values(), ids=FLOOR_MISSES.keys()
)
def test_each_missed_floor_is_named(override: Summary, reason: str) -> None:
    """A value past a floor, or a missing pass_rate, gives one reason."""
    [failure] = floor_failures(_evaluated({**CLEAR, **override}), FLOORS)

    assert failure.startswith(reason)


def test_low_judge_coverage_misses_its_floor() -> None:
    """A judge scoring less than min_judge_coverage of its rows fails the floor."""
    [failure] = floor_failures(_evaluated(CLEAR, coverage={"helpful": 0.5}), FLOORS)

    assert failure == "judge helpful scored 0.5 of the rows due, below 0.9"


def test_judge_coverage_counts_judge_failures_but_not_rows_it_skips() -> None:
    """A judge_error row counts against the judge; a row with no criteria does not."""
    key, comments = golden_settings.keys.criteria, golden_settings.comments
    rows = [
        make_row("scored", {key: True}),
        make_row("failed", {key: None}, comments={key: comments.judge_error}),
        make_row("skipped", {key: None}, comments={key: "no expected.criteria"}),
        make_row("crashed", {}, feedback_errors=frozenset({key})),
    ]

    evaluation = _evaluation(rows, {}, (key,), (key,))

    assert evaluation.judge_coverage == {key: 1 / 3}
    # A judge_error row is unscored, as summarize counts it; coverage catches it.
    assert evaluation.passes == {"scored": True, "crashed": False}
    assert [row.case_id for row in evaluation.failed] == ["crashed"]


ATTACK = "attack_succeeded"
REDTEAM = {
    "within_limit": ({ATTACK: 0.5, f"{ATTACK}_coverage": 1.0}, 0.25, 0.25, None),
    "above_limit": (
        {ATTACK: 0.75, f"{ATTACK}_coverage": 1.0},
        0.25,
        0.25,
        "attack_succeeded 0.75 is above the baseline 0.25 plus 0.25",
    ),
    "unscored_attack": (
        {ATTACK: 0.0, f"{ATTACK}_coverage": 0.5},
        0.0,
        1.0,
        "attack_succeeded scored 0.5 of the red-team rows, not all",
    ),
}


@pytest.mark.parametrize(
    ("summary", "baseline", "limit", "reason"), REDTEAM.values(), ids=REDTEAM.keys()
)
def test_red_team_gate_needs_full_coverage_and_a_bounded_regression(
    summary: Summary, baseline: float, limit: float, reason: str | None
) -> None:
    """Every attack must be scored and the attack rate at most baseline plus limit."""
    failures = redteam_failures(summary, {ATTACK: baseline}, limit)

    assert failures == ([] if reason is None else [reason])


def test_red_team_gate_passes_when_no_red_team_ran() -> None:
    """Without a red-team run there is nothing to gate."""
    assert redteam_failures(None, None, 0.0) == []


def test_deltas_cover_keys_both_summaries_score() -> None:
    """Each shared scored key gets <key>_<suffix>; a None on either side is left out."""
    summary: Summary = {"pass_rate": 0.75, "error_rate": None, "rows": 4.0}
    other: Summary = {"pass_rate": 0.5, "error_rate": 0.0}

    assert deltas(summary, other, "delta_vs_baseline") == {
        "pass_rate_delta_vs_baseline": 0.25
    }
