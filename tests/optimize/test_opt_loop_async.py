"""Tests for the reflective optimizer loop, run offline through the real runner."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.fakes import FailingChatModel, FakeChatModel
from tests.optimize.optimize_fakes import (
    CRASH,
    FAILING_JUDGE,
    FLAKY,
    JUDGE,
    OUTAGE,
    REDTEAM,
    REFLECT,
    UNKNOWN,
    CrashingRunner,
    RuleJudge,
    UploadRecorder,
    agent_folder,
    case,
    cases,
    folder_bytes,
    golden_source,
    keyword_agent,
    leaked,
    models,
    offline_runner,
    optimize_toml,
    outage_at,
    outage_on,
    reflection,
    report,
)
from trajectory_evals.core.agent_config import AgentConfig, AgentConfigError
from trajectory_evals.core.runner import ExperimentRunner
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.golden.settings import settings as golden_settings
from trajectory_evals.optimize import loop, optimize
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
async def test_the_reflection_sees_train_failures_and_never_val_or_holdout(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """Every send carries overrides; no reflection call holds a val or holdout text."""
    rows = [
        *cases("returns", "shipping", splits=("train",)),
        *cases("billing", splits=("val",)),
        *cases("warranty", splits=("holdout",)),
    ]
    budget = "iterations = 2, patience = 3"
    config = agent_folder(tmp_path, optimize_toml(budget=budget))
    agent = keyword_agent()
    named = models(*[reflection("returns shipping guard")] * 2)

    with offline_runner() as runner:
        await optimize(
            config=config,
            source=golden_source(tmp_path, rows),
            agent=agent,
            model_factory=named.__getitem__,
            runner=runner,
            results_dir=tmp_path / "results",
        )

    assert {call.prompt_overrides is not None for call in agent.calls} == {True}
    assert {"billing", "warranty"} <= {call.text for call in agent.calls}
    calls = named[REFLECT].prompts
    assert len(calls) == 2
    [system, first] = calls[0]
    assert system.text == settings.prompts.reflection_system
    assert '<prompt name="system">\nreturns guard\n</prompt>' in first.text
    assert '"input": "shipping"' in first.text
    assert "missing: shipping policy" in first.text
    for _, user in calls:
        assert "billing" not in user.text
        assert "warranty" not in user.text
        assert "train-" not in user.text
    assert network_attempts == []


@pytest.mark.asyncio
async def test_two_accepts_climb_from_the_new_best_and_reset_patience(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """An accept becomes the next parent, has train re-run, and resets patience."""
    config = agent_folder(
        tmp_path, optimize_toml(budget="iterations = 5, patience = 2")
    )
    named = models(
        reflection("returns guard", "Repeat."),
        reflection("returns shipping guard", "Name shipping."),
        reflection("returns shipping guard", "Repeat again."),
        reflection("returns shipping billing guard", "Name billing."),
    )

    loaded = await _optimize(tmp_path, config, named)

    lineage = [
        (record["id"], record["parent"], record["verdict"])
        for record in loaded["candidates"]
    ]
    assert lineage == [
        ("baseline", None, VERDICTS.baseline),
        ("candidate-1", "baseline", VERDICTS.rejected),
        ("candidate-2", "baseline", VERDICTS.accepted),
        ("candidate-3", "candidate-2", VERDICTS.rejected),
        ("candidate-4", "candidate-2", VERDICTS.accepted),
    ]
    # Without the reset, candidate-3 would be the second miss and stop the run.
    assert (loaded["best"], loaded["stop_reason"]) == ("candidate-4", STOPS.no_failures)
    evals = {path.name for path in (loaded["run_dir"] / "evals").iterdir()}
    assert {"baseline-train", "candidate-2-train", "candidate-4-train"} <= evals
    assert not {"candidate-1-train", "candidate-3-train"} & evals
    third = named[REFLECT].prompts[2][1].text
    assert '<prompt name="system">\nreturns shipping guard\n</prompt>' in third
    assert '"input": "billing"' in third
    assert network_attempts == []


FOUR = ("returns", "shipping", "billing", "warranty")
MIN_DELTA = {
    "exactly_min_delta": ("0.25", "returns shipping guard", VERDICTS.accepted),
    "just_below_min_delta": ("0.26", "returns shipping guard", VERDICTS.rejected),
    "tie_at_zero_min_delta": ("0.0", "returns guard", VERDICTS.rejected),
}


@pytest.mark.parametrize(
    ("min_delta", "system", "verdict"), MIN_DELTA.values(), ids=MIN_DELTA.keys()
)
@pytest.mark.asyncio
async def test_min_delta_accepts_its_exact_gain_and_never_a_tie(
    tmp_path: Path,
    network_attempts: list[Any],
    min_delta: str,
    system: str,
    verdict: str,
) -> None:
    """A gain of exactly min_delta passes, less fails, and min_delta 0 needs a gain."""
    budget = f"iterations = 1, min_delta = {min_delta}"
    config = agent_folder(tmp_path, optimize_toml(budget=budget))
    source = golden_source(tmp_path, cases(*FOUR))

    loaded = await _optimize(tmp_path, config, models(reflection(system)), source)

    assert loaded["candidates"][1]["verdict"] == verdict
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


JUDGE_OUTAGES = {
    "baseline_partly_judged": (
        {0},
        "attempts = 2\n",
        "the baseline's attack_succeeded scored 0.5 of the red-team rows",
    ),
    "candidate_judge_error": (
        {1},
        "",
        "attack_succeeded scored 0.0 of the red-team rows",
    ),
}


@pytest.mark.parametrize(
    ("failing", "attempts", "reason"), JUDGE_OUTAGES.values(), ids=JUDGE_OUTAGES.keys()
)
@pytest.mark.asyncio
async def test_a_red_team_judge_outage_blocks_acceptance(
    tmp_path: Path,
    network_attempts: list[Any],
    failing: set[int],
    attempts: str,
    reason: str,
) -> None:
    """An unjudged attack, the baseline's or the candidate's, fails the gate."""
    tables = optimize_toml(redteam=True) + REDTEAM + attempts
    config = agent_folder(tmp_path, tables)
    named = models(reflection("returns shipping guard"))
    named[JUDGE] = RuleJudge(messages=iter(()), rule=outage_at(failing, leaked))

    loaded = await _optimize(tmp_path, config, named)

    candidate = loaded["candidates"][1]
    assert candidate["verdict"] == VERDICTS.rejected
    assert reason in candidate["reason"]
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
        ("agent-optimize-s0-val", "agent-optimize-baseline-val"),
        ("agent-optimize-s0-train", "agent-optimize-baseline-train"),
        ("agent-optimize-s0-val", "agent-optimize-candidate-1-val"),
        ("agent-optimize-s0-holdout", "agent-optimize-baseline-holdout"),
        ("agent-optimize-s0-holdout", "agent-optimize-candidate-1-holdout"),
    ]
    assert network_attempts == []


@pytest.mark.asyncio
async def test_repetitions_count_a_case_fixed_only_when_every_run_passes(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A case that passes one of two runs is not fixed; every send is counted."""
    budget = "iterations = 1, repetitions = 2"
    config = agent_folder(tmp_path, optimize_toml(budget=budget))
    named = models(reflection(f"returns shipping billing guard {FLAKY}-shipping"))

    loaded = await _optimize(tmp_path, config, named)

    candidate = loaded["candidates"][1]
    assert (candidate["fixed_cases"], candidate["regressed_cases"]) == (
        ["val-billing"],
        [],
    )
    assert candidate["val"]["pass_rate"] == pytest.approx(5 / 6)
    assert candidate["val"]["pass_all_k"] == pytest.approx(2 / 3)
    assert candidate["val"]["repetitions"] == 2
    assert candidate["agent_calls"] == 18  # baseline val and train, then this val
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_high_risk_case_left_unscored_blocks_acceptance(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A criteria judge outage on a high-risk case the best passed is a regression."""
    criteria = {"criteria": ["Explains the returns policy."]}
    rows = [
        *cases("shipping", "billing"),
        *(
            {**case("returns", split, risk="high"), "expected": criteria}
            for split in ("train", "val", "holdout")
        ),
    ]
    floors = "pass_rate = 0.0, max_error_rate = 0.0, min_judge_coverage = 0.0"
    config = agent_folder(tmp_path, optimize_toml(floors=floors))
    named = models(reflection("shipping billing guard"))
    criteria_judge = RuleJudge(messages=iter(()), rule=outage_on(UNKNOWN))
    named[golden_settings.judges.model] = criteria_judge

    loaded = await _optimize(tmp_path, config, named, golden_source(tmp_path, rows))

    candidate = loaded["candidates"][1]
    assert candidate["val"]["pass_rate"] == 1.0
    assert candidate["regressed_cases"] == []
    assert candidate["verdict"] == VERDICTS.rejected
    assert candidate["reason"] == "regressed high-risk cases ['val-returns']"
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_judge_outage_on_holdout_blocks_adoption(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A best that wins holdout is not adopted when its judge coverage fails there."""
    rows = [
        *cases(*TOPICS, splits=("train", "val")),
        *({**case(topic, "holdout"), "context": {"note": OUTAGE}} for topic in TOPICS),
    ]
    judge = '[judges.helpful]\nprompt = "inputs_judge.md"\nmodel = "test:rule"\n'
    files = {"inputs_judge.md": "Is this helpful? {inputs} {outputs}"}
    config = agent_folder(tmp_path, optimize_toml() + judge, files=files)
    named = models(reflection("returns shipping guard"))
    named["test:rule"] = RuleJudge(messages=iter(()), rule=outage_on(OUTAGE))

    loaded = await _optimize(tmp_path, config, named, golden_source(tmp_path, rows))

    holdout = loaded["holdout"]
    assert loaded["best"] == "candidate-1"
    assert holdout["best"]["pass_rate"] > holdout["baseline"]["pass_rate"]
    assert loaded["verdict"] == VERDICTS.do_not_adopt
    assert "judge helpful scored 0.0 of the rows due" in loaded["reason"]
    assert network_attempts == []


RED_TEAM_CONFIG = {
    "no_redteam_table": ("", "needs a [redteam] table"),
    "no_policy": ('[redteam]\njudge_model = "test:judge"\n', "needs a policy"),
}


@pytest.mark.parametrize(
    ("tables", "reason"), RED_TEAM_CONFIG.values(), ids=RED_TEAM_CONFIG.keys()
)
@pytest.mark.asyncio
async def test_red_teaming_on_without_a_table_or_policy_fails_before_any_call(
    tmp_path: Path, network_attempts: list[Any], tables: str, reason: str
) -> None:
    """redteam = true must not skip the gate silently or fail every attack later."""
    config = agent_folder(tmp_path, optimize_toml(redteam=True) + tables)
    agent = keyword_agent()

    with offline_runner() as runner, pytest.raises(AgentConfigError) as raised:
        await optimize(
            config=config,
            source=golden_source(tmp_path, cases(*TOPICS)),
            agent=agent,
            model_factory=models().__getitem__,
            runner=runner,
            results_dir=tmp_path / "results",
        )

    assert reason in str(raised.value)
    assert agent.calls == []
    assert not (tmp_path / "results").exists()
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_crash_after_a_candidate_keeps_its_record_in_the_report(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """report.json is rewritten after each candidate, so a holdout crash keeps them."""
    config = agent_folder(tmp_path, optimize_toml())
    named = models(reflection("returns shipping guard"))

    with offline_runner() as runner, pytest.raises(ConnectionError):
        await optimize(
            config=config,
            source=golden_source(tmp_path, cases(*TOPICS)),
            agent=keyword_agent(),
            model_factory=named.__getitem__,
            runner=cast(ExperimentRunner, CrashingRunner(runner, "holdout")),
            results_dir=tmp_path / "results",
        )

    [run_dir] = (tmp_path / "results").iterdir()
    loaded = report(run_dir)
    assert _verdicts(loaded) == [
        ("baseline", VERDICTS.baseline),
        ("candidate-1", VERDICTS.accepted),
    ]
    assert loaded["best"] == "candidate-1"
    assert (loaded["stop_reason"], loaded["holdout"], loaded["verdict"]) == (
        None,
        None,
        None,
    )
    assert network_attempts == []


@pytest.mark.asyncio
async def test_reflection_excerpts_clip_inputs_replies_and_comments(
    tmp_path: Path, network_attempts: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every excerpt field, grader comments included, stops at excerpt_chars."""
    clipped = settings.run.model_copy(update={"excerpt_chars": 6})
    monkeypatch.setattr(loop, "settings", settings.model_copy(update={"run": clipped}))
    config = agent_folder(tmp_path, optimize_toml())
    named = models(reflection("returns shipping guard"))

    await _optimize(tmp_path, config, named)

    user = named[REFLECT].prompts[0][1].text
    failures = json.loads(user.partition("<failures>")[2].partition("</failures>")[0])
    assert sorted(failures, key=lambda excerpt: excerpt["input"]) == [
        {"input": "billin", "reply": "Sorry,", "comments": {"contains": "missin"}},
        {"input": "shippi", "reply": "Sorry,", "comments": {"contains": "missin"}},
    ]
    assert network_attempts == []
