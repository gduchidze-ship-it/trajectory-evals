"""Offline runs of check, eval, optimize and sweep on examples/support_agent."""

import itertools
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from tests.examples.support_agent.support_fakes import (
    SUPPORT_DIR,
    RoutedChatModel,
    support_agent,
)
from tests.examples.support_agent.test_ex_support_golden_async import SCRIPTS
from tests.fakes import FakeChatModel, judge_reply
from tests.optimize.optimize_fakes import folder_bytes, reflection
from trajectory_evals.cli import DEFAULT_BUILDERS, Builders, main
from trajectory_evals.golden.cases import JsonlGoldenSource

GOLDEN = Path("datasets") / "golden.sample.jsonl"
REFLECT = "test:reflect"


def _passing_judge() -> FakeChatModel:
    """Return a judge that passes every call."""
    return FakeChatModel(messages=itertools.repeat(judge_reply(True)))


def _builders(agent_model: BaseChatModel, **models: BaseChatModel) -> Builders:
    """Return builders bridging the support agent on agent_model, models by name.

    A model name not in models gets a judge that always passes.
    """
    judge = _passing_judge()
    return replace(
        DEFAULT_BUILDERS,
        agent=lambda _: support_agent(agent_model),
        model_factory=lambda name: models.get(name, judge),
    )


def test_check_builds_the_real_support_agent_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch, network_attempts: list[Any]
) -> None:
    """The agent's chat model is made on its first turn, so check needs no key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = main(["check", str(SUPPORT_DIR)])

    assert (code, network_attempts) == (0, [])


def test_check_validates_the_support_agent_offline(
    network_attempts: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Every table and sample dataset of the example passes check."""
    code = main(["check", str(SUPPORT_DIR)], builders=_builders(_passing_judge()))

    assert code == 0
    assert capsys.readouterr().out.splitlines()[1:] == [
        "datasets/golden.sample.jsonl: 12 cases",
        "datasets/scenarios.sample.jsonl: 4 cases",
        "tables: ok",
        "card: ok",
    ]
    assert network_attempts == []


def test_eval_grades_the_support_sample_offline(
    tmp_path: Path, network_attempts: list[Any]
) -> None:
    """The first three golden cases run and grade; the lie fails its row."""
    source = JsonlGoldenSource(SUPPORT_DIR / GOLDEN)
    inputs = {case.id: case.input for case in source.cases()}
    model = RoutedChatModel(
        routes={str(inputs[case_id]): script for case_id, script in SCRIPTS.items()}
    )
    results = tmp_path / "results"

    code = main(
        ["eval", str(SUPPORT_DIR), "--data", str(SUPPORT_DIR / GOLDEN), "--local"]
        + ["--limit", "3", "--results-dir", str(results)],
        builders=_builders(model),
    )

    (run_dir,) = results.iterdir()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert code == 0
    assert summary["rows"] == 3
    assert summary["pass_rate"] == pytest.approx(2 / 3)
    assert (summary["tools_called"], summary["criteria"]) == (1.0, 1.0)
    assert network_attempts == []


def _support_copy(tmp_path: Path) -> Path:
    """Copy the example for one fast optimizer round, without the red-team gate.

    Its reflection model gets its own name, so a test can script it.
    """
    folder = tmp_path / "support_agent"
    shutil.copytree(SUPPORT_DIR, folder, ignore=shutil.ignore_patterns("__pycache__"))
    toml = folder / "agent.toml"
    text = toml.read_text(encoding="utf-8")
    text = text.replace("iterations = 3", "iterations = 1").replace(
        'reflection_model = "openai:gpt-5.5"',
        f'reflection_model = "{REFLECT}"\nredteam = false',
    )
    toml.write_text(text, encoding="utf-8")
    return folder


def _canned_agent_model() -> FakeChatModel:
    """Return an agent model that answers every message the same, with no tools."""
    return FakeChatModel(messages=itertools.repeat(AIMessage("Let me check.")))


def test_optimize_writes_a_report_and_leaves_the_prompts_alone(
    tmp_path: Path, network_attempts: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """One reflective round runs offline, reports, and writes no prompt file."""
    folder = _support_copy(tmp_path)
    before = folder_bytes(folder / "prompts")
    reflect = FakeChatModel(messages=iter([reflection("Always look orders up.")]))
    results = tmp_path / "results"

    code = main(
        ["optimize", str(folder), "--data", str(folder / GOLDEN), "--local"]
        + ["--results-dir", str(results)],
        builders=_builders(_canned_agent_model(), **{REFLECT: reflect}),
    )

    (run_dir,) = results.iterdir()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert code == 0
    assert (report["best"], report["verdict"]) == ("baseline", "do not adopt")
    assert [record["id"] for record in report["candidates"]] == [
        "baseline",
        "candidate-1",
    ]
    assert (run_dir / "best" / "system.md").is_file()
    assert folder_bytes(folder / "prompts") == before
    out = capsys.readouterr().out.splitlines()
    assert out[-2].startswith("verdict: do not adopt (")
    assert out[-1] == f"results: {run_dir / 'report.json'}"
    assert network_attempts == []


def test_sweep_prints_each_variant_and_the_verdict(
    tmp_path: Path, network_attempts: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The baseline and the short_prompt variant run on val and are tabled."""
    folder = _support_copy(tmp_path)
    results = tmp_path / "results"

    code = main(
        ["sweep", str(folder), "--data", str(folder / GOLDEN), "--local"]
        + ["--results-dir", str(results)],
        builders=_builders(_canned_agent_model()),
    )

    (run_dir,) = results.iterdir()
    # The SDK prints its own progress lines first; the command's come last.
    *_, baseline, variant, verdict, path = capsys.readouterr().out.splitlines()
    assert code == 0
    assert baseline.startswith("baseline: cost 1.0, val pass_rate ")
    assert variant.startswith("short_prompt: cost 0.7, val pass_rate ")
    assert verdict.startswith("verdict: do not adopt (")
    assert path == f"results: {run_dir / 'report.json'}"
    assert network_attempts == []
