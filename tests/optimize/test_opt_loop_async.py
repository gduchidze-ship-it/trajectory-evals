"""Tests for the reflective optimizer loop, run offline through the real runner."""

from pathlib import Path
from typing import Any, cast

import pytest

from tests.fakes import FailingChatModel, FakeChatModel
from tests.optimize.optimize_fakes import (
    CRASH,
    FAILING_JUDGE,
    REDTEAM,
    REFLECT,
    UploadRecorder,
    agent_folder,
    case,
    cases,
    folder_bytes,
    golden_source,
    keyword_agent,
    models,
    offline_runner,
    optimize_toml,
    reflection,
    report,
)
from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.runner import ExperimentRunner
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.optimize import optimize
from trajectory_evals.optimize.settings import settings

VERDICTS, STOPS = settings.verdicts, settings.stops
TOPICS = ("returns", "shipping", "billing")


async def _optimize(
    tmp_path: Path,
    config: AgentConfig,
    named: dict[str, FakeChatModel],
    source: JsonlGoldenSource | None = None,
) -> dict[str, Any]:
    """Run the optimizer offline on the three topics and return its report."""
    source = source or golden_source(tmp_path, cases(*TOPICS))
    with offline_runner() as runner:
        run_dir = await optimize(
            config=config,
            source=source,
            agent=keyword_agent(),
            model_factory=named.__getitem__,
            runner=runner,
            results_dir=tmp_path / "results",
        )
    loaded = report(run_dir)
    loaded["run_dir"] = run_dir
    return loaded


def _verdicts(loaded: dict[str, Any]) -> list[tuple[str, str]]:
    """Return each candidate's id and verdict."""
    return [(record["id"], record["verdict"]) for record in loaded["candidates"]]


@pytest.mark.asyncio
async def test_a_better_candidate_is_accepted_and_adopted(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A revision naming one more topic wins val and holdout; files stay untouched."""
    config = agent_folder(tmp_path, optimize_toml())
    before = folder_bytes(config.root)
    named = models(reflection("returns shipping guard", "Name shipping."))

    loaded = await _optimize(tmp_path, config, named)

    run_dir: Path = loaded["run_dir"]
    [candidate] = loaded["candidates"][1:]
    assert _verdicts(loaded) == [
        ("baseline", VERDICTS.baseline),
        ("candidate-1", VERDICTS.accepted),
    ]
    assert candidate["parent"] == "baseline"
    assert candidate["rationale"] == "Name shipping."
    assert (candidate["fixed_cases"], candidate["regressed_cases"]) == (
        ["val-shipping"],
        [],
    )
    assert candidate["sign_test_p"] == 0.5
    assert candidate["deltas"]["pass_rate_delta_vs_baseline"] == pytest.approx(1 / 3)
    assert candidate["deltas"]["pass_rate_delta_vs_previous"] == pytest.approx(1 / 3)
    assert (loaded["best"], loaded["stop_reason"]) == ("candidate-1", STOPS.iterations)
    assert loaded["splits"] == {"train": 3, "val": 3, "holdout": 3, "labelled": True}
    holdout = loaded["holdout"]
    assert holdout["baseline"]["pass_rate"] == pytest.approx(1 / 3)
    assert holdout["best"]["pass_rate"] == pytest.approx(2 / 3)
    assert (loaded["verdict"], loaded["reason"]) == (VERDICTS.adopt, None)
    assert (run_dir / "best" / "system.md").read_text() == "returns shipping guard"
    assert (run_dir / "candidates/baseline/system.md").read_text() == "returns guard"
    assert sorted(path.name for path in (run_dir / "evals").iterdir()) == [
        "baseline-holdout",
        "baseline-train",
        "baseline-val",
        "candidate-1-holdout",
        "candidate-1-val",
    ]
    assert folder_bytes(config.root) == before
    assert network_attempts == []


@pytest.mark.asyncio
async def test_the_baseline_is_always_sent_and_failures_reach_the_reflection(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """Every call carries overrides; the reflection sees failures but no case id."""
    config = agent_folder(tmp_path, optimize_toml())
    agent = keyword_agent()
    named = models(reflection("returns shipping guard"))

    with offline_runner() as runner:
        await optimize(
            config=config,
            source=golden_source(tmp_path, cases(*TOPICS)),
            agent=agent,
            model_factory=named.__getitem__,
            runner=runner,
            results_dir=tmp_path / "results",
        )

    assert {call.prompt_overrides is not None for call in agent.calls} == {True}
    [[system, user]] = named[REFLECT].prompts
    assert system.text == settings.prompts.reflection_system
    assert '<prompt name="system">\nreturns guard\n</prompt>' in user.text
    assert '"input": "shipping"' in user.text
    assert "missing: shipping policy" in user.text
    assert "train-shipping" not in user.text
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_candidate_below_a_floor_is_rejected_and_not_adopted(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A gain that misses the pass_rate floor is rejected; the baseline stays best."""
    floors = "pass_rate = 0.9, max_error_rate = 0.0"
    config = agent_folder(tmp_path, optimize_toml(floors=floors))
    named = models(reflection("returns shipping guard"))

    loaded = await _optimize(tmp_path, config, named)

    [candidate] = loaded["candidates"][1:]
    assert candidate["verdict"] == VERDICTS.rejected
    assert "below the floor 0.9" in candidate["reason"]
    assert candidate["redteam"] is None
    assert loaded["best"] == "baseline"
    assert (loaded["verdict"], loaded["reason"]) == (
        VERDICTS.do_not_adopt,
        settings.reasons.is_baseline,
    )
    assert loaded["holdout"]["best"] == loaded["holdout"]["baseline"]
    assert not (loaded["run_dir"] / "evals" / "candidate-1-holdout").exists()
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_judge_that_fails_on_every_row_blocks_acceptance(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """An erroring judge leaves rows unscored, so judge coverage fails the floor."""
    config = agent_folder(tmp_path, optimize_toml() + FAILING_JUDGE)
    named = models(reflection("returns shipping guard"))

    loaded = await _optimize(tmp_path, config, named)

    [candidate] = loaded["candidates"][1:]
    assert candidate["judge_coverage"] == {"helpful": 0.0}
    assert candidate["verdict"] == VERDICTS.rejected
    assert "judge helpful scored 0.0 of the rows due" in candidate["reason"]
    assert network_attempts == []


REDTEAM_CASES = {
    "attack_rate": ("returns shipping", "attack_succeeded 1.0 is above the baseline"),
    "attack_coverage": (
        f"returns shipping guard {CRASH}",
        "attack_succeeded scored 0.0 of the red-team rows",
    ),
}


@pytest.mark.parametrize(
    ("system", "reason"), REDTEAM_CASES.values(), ids=REDTEAM_CASES.keys()
)
@pytest.mark.asyncio
async def test_a_red_team_regression_or_unscored_attack_is_rejected(
    tmp_path: Path, network_attempts: list[Any], system: str, reason: str
) -> None:
    """Dropping the guard raises the attack rate; a crash leaves an attack unscored."""
    config = agent_folder(tmp_path, optimize_toml(redteam=True) + REDTEAM)
    named = models(reflection(system))

    loaded = await _optimize(tmp_path, config, named)

    baseline, candidate = loaded["candidates"]
    assert baseline["redteam"]["attack_succeeded"] == 0.0
    assert candidate["verdict"] == VERDICTS.rejected
    assert reason in candidate["reason"]
    assert loaded["best"] == "baseline"
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_red_teamed_candidate_that_keeps_the_guard_is_accepted(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A candidate whose attack rate matches the baseline's passes the red-team gate."""
    config = agent_folder(tmp_path, optimize_toml(redteam=True) + REDTEAM)
    named = models(reflection("returns shipping guard"))

    loaded = await _optimize(tmp_path, config, named)

    candidate = loaded["candidates"][1]
    assert candidate["verdict"] == VERDICTS.accepted
    assert candidate["redteam"]["attack_succeeded"] == 0.0
    assert candidate["redteam"]["attack_succeeded_coverage"] == 1.0
    assert (loaded["run_dir"] / "evals" / "candidate-1-redteam").is_dir()
    assert network_attempts == []


@pytest.mark.asyncio
async def test_breaking_a_high_risk_case_blocks_acceptance(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A net gain that regresses a high-risk case is rejected, naming the case."""
    rows = [
        *cases("shipping", "billing"),
        *(case("returns", split, risk="high") for split in ("train", "val", "holdout")),
    ]
    config = agent_folder(tmp_path, optimize_toml())
    named = models(reflection("shipping billing guard"))

    loaded = await _optimize(tmp_path, config, named, golden_source(tmp_path, rows))

    [candidate] = loaded["candidates"][1:]
    assert candidate["regressed_cases"] == ["val-returns"]
    assert candidate["fixed_cases"] == ["val-billing", "val-shipping"]
    assert candidate["verdict"] == VERDICTS.rejected
    assert "regressed high-risk cases ['val-returns']" in candidate["reason"]
    assert network_attempts == []


@pytest.mark.asyncio
async def test_patience_stops_after_that_many_misses_in_a_row(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """Two revisions that gain nothing use up a patience of two before iterations."""
    budget = "iterations = 5, patience = 2"
    config = agent_folder(tmp_path, optimize_toml(budget=budget))
    named = models(
        reflection("returns guard", "First idea."),
        reflection("returns guard", "Second idea."),
    )

    loaded = await _optimize(tmp_path, config, named)

    assert _verdicts(loaded)[1:] == [
        ("candidate-1", VERDICTS.rejected),
        ("candidate-2", VERDICTS.rejected),
    ]
    assert loaded["stop_reason"] == STOPS.patience
    [first, second] = named[REFLECT].prompts
    assert '"First idea."' not in first[1].text
    assert '"First idea."' in second[1].text.rpartition("<rejected>")[2]
    assert network_attempts == []


@pytest.mark.asyncio
async def test_the_agent_call_budget_stops_before_any_reflection(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """The baseline's val run spends the budget, so no revision is asked for."""
    budget = "iterations = 5, max_agent_calls = 3"
    config = agent_folder(tmp_path, optimize_toml(budget=budget))
    named = models()

    loaded = await _optimize(tmp_path, config, named)

    assert loaded["stop_reason"] == STOPS.max_agent_calls
    assert _verdicts(loaded) == [("baseline", VERDICTS.baseline)]
    assert named[REFLECT].prompts == []
    assert loaded["agent_calls"] == 6  # val, then holdout
    assert network_attempts == []


@pytest.mark.asyncio
async def test_no_failures_left_on_train_stops_the_loop(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A baseline that passes every train case leaves nothing to reflect on."""
    config = agent_folder(tmp_path, optimize_toml(), system=" ".join(TOPICS))
    named = models()

    loaded = await _optimize(tmp_path, config, named)

    assert loaded["stop_reason"] == STOPS.no_failures
    assert named[REFLECT].prompts == []
    assert network_attempts == []


INVALID = {
    "wrong_names": (
        lambda: models(reflection("returns shipping guard", name="sys")),
        "reflection returned prompts ['sys'], not ['system']",
    ),
    "reflection_error": (
        lambda: {**models(), REFLECT: FailingChatModel(messages=iter(()))},
        "reflection failed: RuntimeError",
    ),
}


@pytest.mark.parametrize(("make", "reason"), INVALID.values(), ids=INVALID.keys())
@pytest.mark.asyncio
async def test_an_unusable_reflection_is_an_invalid_candidate(
    tmp_path: Path, network_attempts: list[Any], make: Any, reason: str
) -> None:
    """Wrong prompt names or a failed reflection count toward patience unevaluated."""
    config = agent_folder(
        tmp_path, optimize_toml(budget="iterations = 3, patience = 1")
    )

    loaded = await _optimize(tmp_path, config, make())

    [candidate] = loaded["candidates"][1:]
    assert (candidate["verdict"], candidate["reason"]) == (VERDICTS.invalid, reason)
    assert candidate["val"] is None
    assert loaded["stop_reason"] == STOPS.patience
    assert not (loaded["run_dir"] / "candidates" / "candidate-1").exists()
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_val_winner_that_does_not_beat_the_baseline_on_holdout_is_not_adopted(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """Holdout asks only about topics both prompts name, so the gain does not carry."""
    rows = [
        *cases(*TOPICS, splits=("train", "val")),
        case("returns", "holdout"),
        case("warranty", "holdout"),
    ]
    config = agent_folder(tmp_path, optimize_toml())
    named = models(reflection("returns shipping billing guard"))

    loaded = await _optimize(tmp_path, config, named, golden_source(tmp_path, rows))

    assert loaded["best"] == "candidate-1"
    assert loaded["holdout"]["best"]["pass_rate"] == 0.5
    assert loaded["verdict"] == VERDICTS.do_not_adopt
    assert loaded["reason"] == settings.reasons.not_above_baseline.format(
        value=0.5, baseline=0.5
    )
    assert network_attempts == []


@pytest.mark.asyncio
async def test_upload_mode_uploads_each_run_to_one_dataset_per_split(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """Every experiment uploads, named for its candidate, on its split's dataset."""
    config = agent_folder(tmp_path, optimize_toml())
    named = models(reflection("returns shipping guard"))

    with offline_runner() as runner:
        recorder = UploadRecorder(runner)
        await optimize(
            config=config,
            source=golden_source(tmp_path, cases(*TOPICS)),
            agent=keyword_agent(),
            model_factory=named.__getitem__,
            runner=cast(ExperimentRunner, recorder),
            results_dir=tmp_path / "results",
            upload=True,
        )

    assert {run.upload_results for _, run in recorder.runs} == {True}
    assert [(dataset, run.experiment_prefix) for dataset, run in recorder.runs] == [
        ("agent-optimize-val", "agent-optimize-baseline-val"),
        ("agent-optimize-train", "agent-optimize-baseline-train"),
        ("agent-optimize-val", "agent-optimize-candidate-1-val"),
        ("agent-optimize-holdout", "agent-optimize-baseline-holdout"),
        ("agent-optimize-holdout", "agent-optimize-candidate-1-holdout"),
    ]
    assert network_attempts == []
