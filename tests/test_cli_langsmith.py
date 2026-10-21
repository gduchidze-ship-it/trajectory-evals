"""Tests for the langsmith pull, push, harvest and score commands."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from langsmith import Client

from tests.e2e.e2e_agent import (
    MODELS_TOML,
    UNBUILDABLE,
    agent_folder,
    one_error_line,
    write_jsonl,
)
from tests.fakes import judge_reply, scripted
from tests.langsmith_io.lsio_fakes import FakeClient, make_run, read_jsonl
from trajectory_evals import cli
from trajectory_evals.cli import Builders, main
from trajectory_evals.langsmith_io import _output

_CASE = {"id": "a1", "input": "Where is A1?", "expected": {"contains": ["A1"]}}
_USER = {"role": "user", "content": "Can you hold order A1?"}
_REPLY = {"role": "assistant", "content": "It is on hold until Friday."}
_PROJECT = '[langsmith]\nproject = "dev-sales"\n'


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the git-ignored data folder, and harvest's default, into tmp_path."""
    data = tmp_path / "data"
    monkeypatch.setattr(_output, "DATA_DIR", data)
    monkeypatch.setattr(cli, "HARVEST_DIR", data / "langsmith")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-fake")
    return data


def _builders(client: FakeClient, **models: Any) -> Builders:
    """Return builders whose uploading client is client; nothing else builds."""
    builders = replace(UNBUILDABLE, client=lambda: cast(Client, client))
    if models:
        builders = replace(builders, model_factory=models.__getitem__)
    return builders


def test_push_then_pull_round_trips_a_golden_file(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """push names the dataset <dataset_prefix>/<stem>; pull writes it back."""
    client = FakeClient()
    folder = agent_folder(tmp_path, '[langsmith]\ndataset_prefix = "support"\n')
    golden = write_jsonl(tmp_path / "golden.sample.jsonl", [_CASE])
    out = data_dir / "pulled.jsonl"

    pushed = main(
        ["langsmith", "push", str(folder), "--data", str(golden)],
        builders=_builders(client),
    )
    pulled = main(
        ["langsmith", "pull", str(folder), "--dataset", "support/golden.sample"]
        + ["--out", str(out)],
        builders=_builders(client),
    )

    assert (pushed, pulled) == (0, 0)
    assert list(client.datasets) == ["support/golden.sample"]
    [row] = read_jsonl(out)
    assert (row["id"], row["expected"]) == ("a1", {"contains": ["A1"]})
    assert capsys.readouterr().out.splitlines() == [
        f"pushed {golden} to dataset support/golden.sample",
        f"wrote {out}",
    ]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["pull", "--dataset", "d", "--out", "pulled.jsonl"], "outside"),
        (["pull", "--dataset", "missing", "--out", "{data}/x.jsonl"], "missing"),
        (["harvest"], "no LangSmith project"),
    ],
)
def test_bad_langsmith_input_exits_2(
    tmp_path: Path,
    data_dir: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    """A tracked output path, an unknown dataset or no project is one line, exit 2."""
    folder = str(agent_folder(tmp_path))
    command = [part.format(data=data_dir) for part in argv]

    code = main(
        ["langsmith", command[0], folder, *command[1:]],
        builders=_builders(FakeClient()),
    )

    assert message in one_error_line(code, capsys)


def test_push_refuses_a_case_still_under_review(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A harvested row nobody reviewed is never pushed."""
    folder = agent_folder(tmp_path)
    golden = write_jsonl(tmp_path / "g.jsonl", [{**_CASE, "review_required": True}])

    code = main(
        ["langsmith", "push", str(folder), "--data", str(golden)],
        builders=_builders(FakeClient()),
    )

    assert "review_required" in one_error_line(code, capsys)


def test_harvest_writes_candidates_under_data_langsmith(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """harvest reads [langsmith] project and writes rows flagged for review."""
    client = FakeClient(
        [make_run({"messages": [_USER]}, {"messages": [_USER, _REPLY]})]
    )
    folder = agent_folder(tmp_path, _PROJECT)

    code = main(
        ["langsmith", "harvest", str(folder), "--limit", "5", "--only-errors"],
        builders=_builders(client),
    )

    [out] = (data_dir / "langsmith").iterdir()
    [row] = read_jsonl(out)
    [query] = client.runs.queries
    assert code == 0
    assert out.name.startswith("shop-")
    assert (row["review_required"], row["observed_output"]) == (True, _REPLY["content"])
    assert (client.projects, query["has_error"]) == (["dev-sales"], True)
    assert capsys.readouterr().out == f"wrote {out}\n"


def test_score_dry_run_prints_counts_and_writes_no_feedback(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """score grades runs with the policy judge and, dry, writes nothing."""
    client = FakeClient(
        [make_run({"messages": [_USER]}, {"messages": [_USER, _REPLY]})]
    )
    folder = agent_folder(tmp_path, MODELS_TOML + _PROJECT)
    judge = scripted(judge_reply(False))

    code = main(
        ["langsmith", "score", str(folder), "--project", "prod", "--dry-run"],
        builders=_builders(client, **{"judge:scripted": judge}),
    )

    assert code == 0
    assert (client.projects, client.feedback) == (["prod"], [])
    assert capsys.readouterr().out.splitlines() == [
        "runs read: 1",
        "policy_violation: scored 1, skipped 0, errors 0, written 0",
    ]
    assert len(judge.prompts) == 1
