"""Tests for the floor-gated variant sweep, run offline through the real runner."""

from pathlib import Path
from typing import Any

import pytest

from tests.optimize.optimize_fakes import (
    REDTEAM,
    agent_folder,
    cases,
    folder_bytes,
    golden_source,
    keyword_agent,
    models,
    offline_runner,
    optimize_toml,
    report,
)
from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.optimize import sweep
from trajectory_evals.optimize.settings import settings

VERDICTS = settings.verdicts
FLOORS = "pass_rate = 0.5, max_error_rate = 0.0"


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
    tmp_path: Path, config: AgentConfig, include: Path | None = None
) -> dict[str, Any]:
    """Run the sweep offline over three topics and return its report."""
    named = models()
    with offline_runner() as runner:
        run_dir = await sweep(
            config=config,
            source=golden_source(tmp_path, cases("returns", "shipping", "billing")),
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
