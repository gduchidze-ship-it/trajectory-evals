"""Tests for the floor-gated variant sweep, run offline through the real runner."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from tests.fakes import FakeChatModel
from tests.optimize.optimize_fakes import (
    JUDGE,
    REDTEAM,
    CrashingRunner,
    RuleJudge,
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
    report,
)
from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.runner import ExperimentRunner
from trajectory_evals.golden.graders import Graders
from trajectory_evals.optimize import sweep
from trajectory_evals.optimize.settings import settings

VERDICTS = settings.verdicts
FLOORS = "pass_rate = 0.5, max_error_rate = 0.0"
TOPICS = ("returns", "shipping", "billing")


def _variant(name: str, cost: float) -> str:
    """Return a [variants.<name>] table overriding system with variants/<name>.md."""
    return (
        f"[variants.{name}]\n"
        f'prompt_overrides = {{ system = "variants/{name}.md" }}\n'
        f"cost = {cost}\n"
    )


def _folder(
    tmp_path: Path, variants: dict[str, tuple[str, float]], tables: str = ""
) -> AgentConfig:
    """Load an agent whose variants map name to (system prompt, cost)."""
    toml = optimize_toml(floors=FLOORS, redteam=bool(tables)) + tables
    toml += "".join(_variant(name, cost) for name, (_, cost) in variants.items())
    files = {f"variants/{name}.md": text for name, (text, _) in variants.items()}
    return agent_folder(tmp_path, toml, files=files)


async def _sweep(
    tmp_path: Path,
    config: AgentConfig,
    include: Path | None = None,
    *,
    named: dict[str, FakeChatModel] | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the sweep offline, over the three topics by default; return its report."""
    named = named or models()
    with offline_runner() as runner:
        run_dir = await sweep(
            config=config,
            source=golden_source(tmp_path, rows or cases(*TOPICS)),
            agent=keyword_agent(),
            model_factory=named.__getitem__,
            runner=runner,
            results_dir=tmp_path / "results",
            include=include,
        )
    return report(run_dir)


def _table(loaded: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """Return each variant's verdict and val pass_rate by id."""
    return {
        row["id"]: (row["verdict"], row["val"]["pass_rate"])
        for row in loaded["variants"]
    }


@pytest.mark.asyncio
async def test_the_cheapest_variant_clearing_every_floor_wins(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """Below-floor and attack-prone variants lose to the cheapest one that passes."""
    config = _folder(
        tmp_path,
        {
            "terse": ("nothing", 0.1),
            "open": ("returns shipping", 0.2),
            "cheap": ("returns shipping guard", 0.4),
            "full": ("returns shipping billing guard", 0.9),
        },
        REDTEAM,
    )
    before = folder_bytes(config.root)

    loaded = await _sweep(tmp_path, config)

    table = _table(loaded)
    assert table["baseline"] == (VERDICTS.rejected, pytest.approx(1 / 3))
    assert table["terse"] == (VERDICTS.rejected, 0.0)
    assert table["open"] == (VERDICTS.rejected, pytest.approx(2 / 3))
    assert table["cheap"] == (VERDICTS.passed, pytest.approx(2 / 3))
    assert table["full"] == (VERDICTS.passed, 1.0)
    rows = {row["id"]: row for row in loaded["variants"]}
    assert rows["baseline"]["redteam"]["attack_succeeded"] == 0.0
    assert rows["terse"]["redteam"] is None
    assert "attack_succeeded 1.0 is above the baseline 0.0" in rows["open"]["reason"]
    cheap = rows["cheap"]
    assert cheap["deltas"]["pass_rate_delta_vs_baseline"] == pytest.approx(1 / 3)
    assert cheap["fixed_cases"] == ["val-shipping"]
    # delta_vs_previous follows variant order: full comes right after cheap.
    assert "pass_rate_delta_vs_previous" not in rows["baseline"]["deltas"]
    assert cheap["deltas"]["pass_rate_delta_vs_previous"] == 0.0
    full = rows["full"]["deltas"]["pass_rate_delta_vs_previous"]
    assert full == pytest.approx(1 / 3)
    assert loaded["winner"] == "cheap"
    assert loaded["holdout"]["best"]["pass_rate"] == pytest.approx(2 / 3)
    assert (loaded["verdict"], loaded["reason"]) == (VERDICTS.adopt, None)
    assert folder_bytes(config.root) == before
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_cost_tie_goes_to_the_higher_pass_rate_and_include_competes(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """At equal cost the better variant wins; an included folder costs as baseline."""
    best = tmp_path / "previous" / "best"
    best.mkdir(parents=True)
    (best / "system.md").write_text("returns shipping billing", encoding="utf-8")
    config = _folder(
        tmp_path,
        {"two": ("returns shipping", 0.5), "three": ("returns shipping billing", 0.5)},
    )

    loaded = await _sweep(tmp_path, config, include=best)

    included = loaded["variants"][-1]
    assert (included["id"], included["cost"]) == ("include", 1.0)
    assert included["val"]["pass_rate"] == 1.0
    assert loaded["winner"] == "three"
    assert network_attempts == []


@pytest.mark.asyncio
async def test_no_variant_clearing_the_floors_means_no_holdout_run(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """With every candidate below a floor there is no winner and nothing to adopt."""
    config = _folder(tmp_path, {"terse": ("nothing", 0.1)})

    loaded = await _sweep(tmp_path, config)

    assert (loaded["winner"], loaded["holdout"]) == (None, None)
    assert (loaded["verdict"], loaded["reason"]) == (
        VERDICTS.do_not_adopt,
        settings.reasons.no_winner,
    )
    assert network_attempts == []


def _claimed_pass_rate(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Report a pass_rate of 1 whatever the rows scored."""
    return {"results": [{"key": "pass_rate", "score": 1}]}


def claims_every_pass(_config: AgentConfig) -> Graders:
    """Return Graders whose summary evaluator claims summarize's pass_rate key."""
    return Graders(summary_evaluators=(_claimed_pass_rate,))


@pytest.mark.asyncio
async def test_a_graders_pass_rate_cannot_lift_a_candidate_over_the_floor(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """The floors gate on summarize's pass_rate, never a [graders] key of that name."""
    graders = (
        '[graders]\nfactory = "tests.optimize.test_opt_sweep_async:claims_every_pass"\n'
    )
    toml = optimize_toml(floors=FLOORS) + graders + _variant("terse", 0.1)
    config = agent_folder(tmp_path, toml, files={"variants/terse.md": "nothing"})

    loaded = await _sweep(tmp_path, config)

    assert _table(loaded) == {
        "baseline": (VERDICTS.rejected, pytest.approx(1 / 3)),
        "terse": (VERDICTS.rejected, 0.0),
    }
    assert loaded["winner"] is None
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_baseline_winner_is_not_adopted(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """When the baseline is the cheapest passing candidate, nothing changes."""
    config = _folder(tmp_path, {"pricey": ("returns shipping billing", 2.0)})
    system = config.root / "prompts" / "system.md"
    system.write_text("returns shipping", encoding="utf-8")

    loaded = await _sweep(tmp_path, config)

    assert loaded["winner"] == "baseline"
    assert loaded["holdout"]["best"] == loaded["holdout"]["baseline"]
    assert (loaded["verdict"], loaded["reason"]) == (
        VERDICTS.do_not_adopt,
        settings.reasons.is_baseline,
    )
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_variant_breaking_a_high_risk_case_cannot_win(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """The cheapest variant clears the floors but breaks a high-risk case, so loses."""
    rows = [
        *cases("shipping", "billing"),
        *(case("returns", split, risk="high") for split in ("train", "val", "holdout")),
    ]
    config = _folder(
        tmp_path,
        {"risky": ("shipping billing", 0.1), "safe": ("returns shipping", 0.5)},
    )

    loaded = await _sweep(tmp_path, config, rows=rows)

    table = _table(loaded)
    assert table["risky"] == (VERDICTS.rejected, pytest.approx(2 / 3))
    [risky] = [row for row in loaded["variants"] if row["id"] == "risky"]
    assert risky["reason"] == "regressed high-risk cases ['val-returns']"
    assert loaded["winner"] == "safe"
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_partly_judged_baseline_red_team_blocks_every_variant(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """A baseline attack rate measured on fewer attacks gates nothing, so none wins."""
    config = _folder(
        tmp_path, {"cheap": ("returns shipping guard", 0.4)}, REDTEAM + "attempts = 2\n"
    )
    named = models()
    named[JUDGE] = RuleJudge(messages=iter(()), rule=outage_at({0}, leaked))

    loaded = await _sweep(tmp_path, config, named=named)

    [_, cheap] = loaded["variants"]
    assert cheap["redteam"]["attack_succeeded_coverage"] == 1.0
    assert "the baseline's attack_succeeded scored 0.5" in cheap["reason"]
    assert (loaded["winner"], loaded["verdict"]) == (None, VERDICTS.do_not_adopt)
    assert network_attempts == []


@pytest.mark.asyncio
async def test_a_crash_after_a_variant_keeps_its_row_in_the_report(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """report.json is rewritten after each variant, so a holdout crash keeps them."""
    config = _folder(tmp_path, {"cheap": ("returns shipping guard", 0.4)})

    with offline_runner() as runner, pytest.raises(ConnectionError):
        await sweep(
            config=config,
            source=golden_source(tmp_path, cases(*TOPICS)),
            agent=keyword_agent(),
            model_factory=models().__getitem__,
            runner=cast(ExperimentRunner, CrashingRunner(runner, "holdout")),
            results_dir=tmp_path / "results",
        )

    [run_dir] = (tmp_path / "results").iterdir()
    loaded = report(run_dir)
    assert [row["id"] for row in loaded["variants"]] == ["baseline", "cheap"]
    assert (loaded["winner"], loaded["verdict"]) == (None, None)
    assert network_attempts == []
