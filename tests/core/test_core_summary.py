"""Tests for the scores summarize computes over an experiment's rows."""

import pytest

from tests.core.core_fakes import make_row
from trajectory_evals.core.summary import summarize


def test_feedback_means_skip_unscored_values_and_report_coverage() -> None:
    """A key's mean is over its scored rows, bools as 0 or 1, beside its coverage."""
    rows = [
        make_row("a", {"correct": True, "judge": 0.5}),
        make_row("b", {"correct": False, "judge": None}),
        make_row("c", {"correct": None}),
    ]

    summary = summarize(rows)

    assert summary["correct"] == 0.5
    assert summary["correct_coverage"] == pytest.approx(2 / 3)
    assert summary["judge"] == 0.5
    assert summary["judge_coverage"] == pytest.approx(1 / 3)
    assert (summary["pass_rate"], summary["pass_rate_ci_low"]) == (None, None)


def test_a_deterministic_fail_beats_a_judge_pass() -> None:
    """A row passes only if every pass key it scored passed; unscored rows are out."""
    rows = [
        make_row("fails", {"contains": False, "criteria": True}),
        make_row("passes", {"contains": True, "criteria": True}),
        make_row("judged", {"contains": None, "criteria": True}),
        make_row("unscored", {"contains": None, "criteria": None}),
    ]

    summary = summarize(rows, pass_keys=("contains", "criteria"))

    assert summary["pass_rate"] == pytest.approx(2 / 3)


def test_pass_rate_has_a_wilson_interval() -> None:
    """8 passes in 10 rows give the Wilson 95% interval of about 0.490 to 0.943."""
    rows = [make_row(f"case-{index}", {"ok": index < 8}) for index in range(10)]

    summary = summarize(rows, pass_keys=("ok",))

    assert summary["pass_rate"] == 0.8
    assert summary["pass_rate_ci_low"] == pytest.approx(0.4902, abs=1e-4)
    assert summary["pass_rate_ci_high"] == pytest.approx(0.9433, abs=1e-4)


def test_repeated_cases_report_pass_all_and_pass_any() -> None:
    """A case passes all only if every repetition passed; one run per case omits it."""
    runs = {"steady": [True, True, True], "flaky": [True, False, True]}
    rows = [
        make_row(case_id, {"ok": passed})
        for case_id, passes in runs.items()
        for passed in passes
    ]

    repeated = summarize(rows, pass_keys=("ok",))
    single = summarize(rows[::3], pass_keys=("ok",))

    assert (repeated["pass_all_k"], repeated["pass_any_k"]) == (0.5, 1.0)
    assert (repeated["repetitions"], repeated["pass_rate"]) == (3, pytest.approx(5 / 6))
    assert {"pass_all_k", "pass_any_k", "repetitions"}.isdisjoint(single)


def test_errors_and_latency_percentiles() -> None:
    """error_rate counts every row; latency percentiles skip rows without one."""
    timed = [make_row(f"case-{ms}", latency_ms=float(ms)) for ms in range(10, 101, 10)]
    errored = make_row("errored", error="RuntimeError('boom')")

    summary = summarize([*timed, errored])

    assert summary["error_rate"] == pytest.approx(1 / 11)
    assert summary["latency_p50_ms"] == 55.0
    assert summary["latency_p95_ms"] == pytest.approx(95.5)
    assert summary["rows"] == 11


def test_segments_read_metadata_before_outputs() -> None:
    """A segment value comes from metadata, else outputs; rows with neither drop out."""
    rows = [
        make_row(
            "a",
            {"ok": True},
            metadata={"persona": "crescendo"},
            outputs={"persona": "ignored"},
        ),
        make_row("b", {"ok": False}, outputs={"persona": "roleplay"}),
        make_row("c", {"ok": False}, metadata={"persona": "crescendo"}),
        make_row("d", {"ok": True}),
        make_row(
            "e",
            {"ok": True},
            metadata={"persona": None},
            outputs={"persona": "roleplay"},
        ),
    ]

    summary = summarize(rows, pass_keys=("ok",), segment_by=("persona",))

    assert {key: value for key, value in summary.items() if "[" in key} == {
        "pass_rate[persona=crescendo]": 0.5,
        "pass_rate[persona=roleplay]": 0.5,
    }


def test_errored_rows_and_crashed_graders_fail() -> None:
    """With pass keys, a raised target or a crashed grader fails even a judge pass."""
    passing = [make_row(f"pass-{index}", {"ok": True}) for index in range(6)]
    errored = [make_row(f"error-{index}", error="boom") for index in range(3)]
    crashed = make_row(
        "crashed", {"judge": True, "wrapper": None}, feedback_errors={"wrapper"}
    )

    summary = summarize([*passing, *errored, crashed], pass_keys=("ok", "judge"))
    repeated = summarize(
        [make_row("a", {"ok": True}), make_row("a", error="boom")], pass_keys=("ok",)
    )

    assert (summary["pass_rate"], summary["error_rate"]) == (0.6, 0.3)
    assert (repeated["pass_all_k"], repeated["pass_any_k"]) == (0.0, 1.0)
    assert summarize(errored)["pass_rate"] is None


def test_only_a_score_of_one_passes() -> None:
    """A pass key passes at 1 or True; a continuous 0.2 is a fail, not a pass."""
    rows = [make_row(str(score), {"ok": score}) for score in (0.2, 1.0, 1, True)]

    assert summarize(rows, pass_keys=("ok",))["pass_rate"] == 0.75


def test_repetitions_do_not_narrow_the_wilson_interval() -> None:
    """Three runs of ten cases give the interval of the ten cases, not of 30 rows."""
    rows = [
        make_row(f"case-{index}", {"ok": index < 8})
        for index in range(10)
        for _ in range(3)
    ]

    summary = summarize(rows, pass_keys=("ok",))

    assert summary["pass_rate_ci_low"] == pytest.approx(0.4902, abs=1e-4)
    assert summary["pass_rate_ci_high"] == pytest.approx(0.9433, abs=1e-4)


def test_feedback_keys_come_sorted() -> None:
    """Feedback keys are reported in key order, whatever order the rows hold them."""
    summary = summarize([make_row("a", {"zeta": 1, "alpha": 0})])

    assert list(summary)[:4] == ["alpha", "alpha_coverage", "zeta", "zeta_coverage"]


@pytest.mark.parametrize(
    "feedback",
    [{"pass_rate": 1.0}, {"rows": 1.0}, {"ok": 1.0, "ok_coverage": 1.0}],
    ids=["pass_rate", "rows", "coverage"],
)
def test_a_feedback_key_named_like_a_summary_key_is_refused(
    feedback: dict[str, float],
) -> None:
    """A feedback key that summarize also reports would be overwritten, so it raises."""
    with pytest.raises(ValueError, match="collide"):
        summarize([make_row("a", feedback)])


def test_no_rows_give_no_scores() -> None:
    """With no rows every score is None and no feedback key appears."""
    assert summarize([], pass_keys=("ok",), segment_by=("persona",)) == {
        "pass_rate": None,
        "pass_rate_ci_low": None,
        "pass_rate_ci_high": None,
        "error_rate": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "rows": 0,
    }
