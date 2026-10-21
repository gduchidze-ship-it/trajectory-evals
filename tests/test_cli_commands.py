"""Tests for the CLI's parsing, preflight, bad-input exits and run wiring."""

import json
import re
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

from tests.core.test_core_dataset import FakeLangSmithClient
from tests.e2e.e2e_agent import (
    AGENT_TOML,
    MODELS_TOML,
    RECEIVED,
    UNBUILDABLE,
    agent_folder,
    one_error_line,
    write_jsonl,
)
from tests.fakes import judge_reply, scripted
from trajectory_evals.cli import DEFAULT_BUILDERS, build_parser, main
from trajectory_evals.core import runner
from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.dataset import LangSmithDatasetSync
from trajectory_evals.core.settings import REPO_ROOT
from trajectory_evals.core.settings import settings as core_settings
from trajectory_evals.golden.graders import Graders

_CASE = {"id": "a1", "input": "Where is A1?"}
_JUDGED = {"id": "a2", "input": "Is A1 late?", "expected": {"criteria": ["Answers"]}}
_SCENARIO = {"id": "s1", "persona": "benign_curious", "scenario": "Ask about A1."}
_EXPERIMENT = "uploaded-1"
_SECRET = "sk-live-0123456789-FAKE"


@pytest.fixture(autouse=True)
def received() -> Iterator[list[str]]:
    """Clear the test agent's record of messages before and after each test."""
    RECEIVED.clear()
    yield RECEIVED
    RECEIVED.clear()


def test_help_runs_as_a_module() -> None:
    """python -m trajectory_evals --help lists every command and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "trajectory_evals", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for command in ("check", "serve", "eval", "simulate", "redteam", "optimize"):
        assert command in result.stdout
    assert "langsmith" in result.stdout


def test_eval_defaults() -> None:
    """eval runs uploaded, 4 at once, once each, segmented by risk and difficulty."""
    args = build_parser().parse_args(["eval", "shop", "--data", "g.jsonl"])

    assert (args.agent, args.data) == ("shop", Path("g.jsonl"))
    assert (args.local, args.limit, args.prefix) == (False, None, None)
    assert (args.concurrency, args.repetitions) == (4, 1)
    assert args.results_dir == Path("results")
    assert args.segment_by == ("risk", "difficulty")


def test_redteam_defaults_defer_to_agent_toml() -> None:
    """redteam reads no data, defers personas and attempts, and segments by persona."""
    args = build_parser().parse_args(["redteam", "shop"])

    assert (args.data, args.personas, args.attempts, args.max_turns) == (
        None,
        None,
        None,
        None,
    )
    assert args.segment_by == ("persona",)


def test_personas_are_comma_separated_and_stripped() -> None:
    """--personas splits on commas, strips each name and drops blanks."""
    args = build_parser().parse_args(["redteam", "shop", "--personas", " a, b,,c "])

    assert args.personas == ["a", "b", "c"]


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "shop", "--data", "g.jsonl", "--limit"],
        ["eval", "shop", "--data", "g.jsonl", "--concurrency"],
        ["simulate", "shop", "--scenarios", "s.jsonl", "--max-turns"],
        ["redteam", "shop", "--attempts"],
        ["serve", "shop", "--port"],
        ["langsmith", "harvest", "shop", "--since-days"],
    ],
)
@pytest.mark.parametrize("value", ["0", "-3", "two"])
def test_counts_below_one_are_rejected(
    argv: list[str], value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every count option refuses anything but a whole number of at least one."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([*argv[:-1], f"{argv[-1]}={value}"])

    assert exit_info.value.code == 2
    assert "expected a whole number >= 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "shop"],
        ["simulate", "shop"],
        ["optimize", "shop"],
        ["optimize", "shop", "--data", "g.jsonl", "--local", "--upload-as-experiments"],
        ["langsmith", "pull", "shop", "--dataset", "d"],
        ["compare", "shop"],
    ],
)
def test_usage_errors_exit_2(argv: list[str]) -> None:
    """A missing required option, clashing modes or an unknown command exit 2."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(argv)

    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "shop", "--data", "g.jsonl"],
        ["langsmith", "score", "shop", "--dry-run"],
    ],
)
def test_a_run_reaching_langsmith_without_its_key_exits_2(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --local, an unset LANGSMITH_API_KEY stops the run before any build."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    code = main(argv, builders=UNBUILDABLE)

    assert "LANGSMITH_API_KEY is not set" in one_error_line(code, capsys)


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "shop", "--data", "missing.jsonl", "--local"],
        ["simulate", "shop", "--scenarios", "missing.jsonl", "--local"],
        ["optimize", "shop", "--data", "missing.jsonl", "--local"],
    ],
)
def test_missing_data_file_exits_2_before_any_build(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """A data file that does not exist stops the run before the agent is loaded."""
    code = main(argv, builders=UNBUILDABLE)

    assert "data file not found: missing.jsonl" in one_error_line(code, capsys)


def test_unknown_agent_folder_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An agent path with no agent.toml is reported, not raised."""
    code = main(["redteam", str(tmp_path / "nobody"), "--local"], builders=UNBUILDABLE)

    assert "file not found" in one_error_line(code, capsys)


def test_bad_row_exits_2_without_echoing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad golden row names path:line and its field, never its text."""
    folder = agent_folder(tmp_path)
    secret = "my card number is 4111"
    data = write_jsonl(tmp_path / "g.jsonl", [_CASE, {"id": "b", "inputs": secret}])
    argv = ["eval", str(folder), "--data", str(data), "--local"]

    code = main(argv, builders=UNBUILDABLE)

    error = one_error_line(code, capsys)
    assert f"{data}:2:" in error
    assert secret not in error


@pytest.mark.parametrize(
    ("command", "option"), [("eval", "--data"), ("simulate", "--scenarios")]
)
def test_empty_data_file_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, option: str
) -> None:
    """A data file with no cases stops the run instead of running nothing."""
    folder = agent_folder(tmp_path)
    data = tmp_path / "empty.jsonl"
    data.write_text("\n  \n", encoding="utf-8")
    argv = [command, str(folder), option, str(data), "--local"]

    code = main(argv, builders=UNBUILDABLE)

    assert f"no cases to run in {data}" in one_error_line(code, capsys)


def test_simulate_with_an_unknown_persona_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A scenario naming a persona the agent lacks stops the run before a build."""
    folder = agent_folder(tmp_path)
    ghost = {**_SCENARIO, "id": "s2", "persona": "ghost"}
    scenarios = write_jsonl(tmp_path / "scenarios.jsonl", [_SCENARIO, ghost])
    argv = ["simulate", str(folder), "--scenarios", str(scenarios), "--local"]

    code = main(argv, builders=UNBUILDABLE)

    assert "unknown personas ['ghost']" in one_error_line(code, capsys)


@pytest.mark.parametrize(
    ("tables", "argv", "message"),
    [
        (MODELS_TOML, ["--personas", "benign_curious"], "must be attack personas"),
        ("", [], "needs a [redteam] policy"),
    ],
)
def test_redteam_bad_input_exits_2_before_any_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tables: str,
    argv: list[str],
    message: str,
) -> None:
    """A benign --personas name, or no [redteam] policy, stops the run up front."""
    folder = agent_folder(tmp_path, tables)

    code = main(["redteam", str(folder), "--local", *argv], builders=UNBUILDABLE)

    assert message in one_error_line(code, capsys)


def _sequence_runner_folder(tmp_path: Path) -> Path:
    """Write an agent whose sequence-runner options carry a secret in a bad key."""
    folder = agent_folder(tmp_path, "")
    bridge = (
        '[bridge]\nresponder = "trajectory_evals.connect.sequence_runner:factory"\n'
        'options = { url = "http://runner.test", sequence_id = "s", client_key = "c",'
        ' product_id = "p", messages_key = "messages", reply_path = "reply",'
        f' api_key = "{_SECRET}" }}\n'
    )
    toml = AGENT_TOML.split("[bridge]")[0] + bridge
    (folder / "agent.toml").write_text(toml, encoding="utf-8")
    return folder


@pytest.mark.parametrize("command", ["check", "eval"])
def test_bad_bridge_options_never_print_their_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    """Invalid [bridge] options exit 2 naming the key, never its secret value."""
    folder = _sequence_runner_folder(tmp_path)
    data = write_jsonl(tmp_path / "g.jsonl", [_CASE])
    argv = {"check": [], "eval": ["--data", str(data), "--local"]}[command]

    code = main([command, str(folder), *argv])

    output = capsys.readouterr()
    assert (code, output.err.count("\n")) == (2, 1)
    assert "api_key: Extra inputs are not permitted" in output.err
    assert _SECRET not in output.out + output.err


def test_a_model_that_cannot_be_built_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing OPENAI_API_KEY for the criteria judge is one line, exit 2."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    folder = agent_folder(tmp_path)
    data = write_jsonl(tmp_path / "g.jsonl", [_JUDGED])

    code = main(["eval", str(folder), "--data", str(data), "--local"])

    error = one_error_line(code, capsys)
    assert "cannot build model 'openai:gpt-5.5'" in error
    assert "OPENAI_API_KEY" in error


class _ModelFactory:
    """Records each model name asked for and hands out a passing judge."""

    def __init__(self) -> None:
        """Start with no model asked for."""
        self.names: list[str] = []

    def __call__(self, name: str) -> BaseChatModel:
        """Record name and return a judge that passes once."""
        self.names.append(name)
        return scripted(judge_reply(True))


@pytest.mark.parametrize(
    ("argv", "names"),
    [
        (["--criteria-model", "test:judge"], ["test:judge"]),
        (["--limit", "1"], []),
    ],
)
def test_the_criteria_judge_is_built_only_for_cases_that_run(
    tmp_path: Path,
    network_attempts: list[Any],
    argv: list[str],
    names: list[str],
) -> None:
    """A case with criteria builds its judge, unless --limit leaves the case out."""
    folder = agent_folder(tmp_path)
    data = write_jsonl(tmp_path / "g.jsonl", [_CASE, _JUDGED])
    factory = _ModelFactory()
    results = tmp_path / "results"

    code = main(
        ["eval", str(folder), "--data", str(data), "--local"]
        + ["--results-dir", str(results), *argv],
        builders=replace(DEFAULT_BUILDERS, model_factory=factory),
    )

    (run_dir,) = results.iterdir()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert (code, factory.names) == (0, names)
    assert summary.get("criteria") == (1.0 if names else None)
    assert network_attempts == []


class _RecordingClient:
    """A LangSmith client whose datasets start empty and record their names."""

    def __init__(self) -> None:
        """Start with no datasets created."""
        self.created: list[str] = []

    def read_dataset(self, *, dataset_name: str) -> NoReturn:
        """Report that no dataset exists yet."""
        raise LangSmithNotFoundError(dataset_name)

    def create_dataset(self, dataset_name: str) -> SimpleNamespace:
        """Record the dataset's name and return it with a stable id."""
        self.created.append(dataset_name)
        return SimpleNamespace(id=uuid5(NAMESPACE_URL, dataset_name))

    def list_examples(self, *, dataset_id: UUID) -> list[Any]:
        """Return no stored examples."""
        return []

    def create_examples(
        self, *, dataset_id: UUID, examples: Sequence[dict[str, Any]]
    ) -> None:
        """Accept the examples the sync creates."""


class _FakeResults:
    """The parts of aevaluate's results the runner reads, with no rows."""

    experiment_name = _EXPERIMENT

    async def get_comparison_url(self) -> str:
        """Return the experiment's URL."""
        return "https://smith.example/compare"

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """Yield no rows; these tests check only how aevaluate was called."""
        rows: tuple[dict[str, Any], ...] = ()
        for row in rows:
            yield row


@pytest.mark.parametrize(
    ("command", "tables", "dataset"),
    [
        ("eval", "", "shop/golden"),
        ("eval", '[langsmith]\ndataset_prefix = "support"\n', "support/golden"),
        ("simulate", "", "shop/scenarios"),
        ("redteam", "", r"shop/redteam-[0-9a-f]{8}"),
    ],
)
def test_uploaded_runs_sync_the_named_dataset_on_the_uploading_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    tables: str,
    dataset: str,
) -> None:
    """Without --local the cases sync to <prefix>/<source> and the run uploads.

    A red-team source is named for its persona set, so its dataset is too.
    """
    calls: list[dict[str, Any]] = []

    async def aevaluate(target: Any, /, **kwargs: Any) -> _FakeResults:
        """Record the call and return results with no rows."""
        calls.append(kwargs)
        return _FakeResults()

    monkeypatch.setattr(runner, "aevaluate", aevaluate)
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-fake")
    client = _RecordingClient()
    folder = agent_folder(tmp_path, MODELS_TOML + tables)
    golden = write_jsonl(tmp_path / "golden.jsonl", [_CASE])
    scenarios = write_jsonl(tmp_path / "scenarios.jsonl", [_SCENARIO])
    data = {
        "eval": ["--data", str(golden)],
        "simulate": ["--scenarios", str(scenarios)],
        "redteam": [],
    }[command]
    builders = replace(
        DEFAULT_BUILDERS,
        client=lambda: cast(Client, client),
        model_factory=lambda _: scripted(),
    )

    code = main(
        [command, str(folder), *data, "--results-dir", str(tmp_path / "results")],
        builders=builders,
    )

    [call] = calls
    [created] = client.created
    assert code == 0
    assert re.fullmatch(dataset, created)
    assert (call["data"], call["client"], call["upload_results"]) == (
        created,
        client,
        True,
    )
    assert call["experiment_prefix"] == f"shop-{command}"
    assert (call["metadata"]["agent"], call["metadata"]["command"]) == (
        "shop",
        command,
    )
    assert f"experiment: {_EXPERIMENT}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "namespace", [None, "custom.example.com"], ids=["default", "override"]
)
def test_an_uploaded_eval_and_a_push_give_each_case_the_same_example_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, namespace: str | None
) -> None:
    """Both mirror a file into <prefix>/<stem> under one namespace, so ids agree.

    The namespace is [langsmith].namespace, else core's example_namespace.
    """

    async def aevaluate(target: Any, /, **kwargs: Any) -> _FakeResults:
        """Return results with no rows; only the dataset sync is under test."""
        return _FakeResults()

    monkeypatch.setattr(runner, "aevaluate", aevaluate)
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-fake")
    table = "" if namespace is None else f'[langsmith]\nnamespace = "{namespace}"\n'
    folder = agent_folder(tmp_path, table)
    other = {"id": "a2", "input": "Is A1 late?"}
    golden = write_jsonl(tmp_path / "golden.jsonl", [_CASE, other])
    evaluated, pushed = FakeLangSmithClient(), FakeLangSmithClient()
    results = ["--results-dir", str(tmp_path / "results")]

    eval_code = main(
        ["eval", str(folder), "--data", str(golden), *results],
        builders=replace(DEFAULT_BUILDERS, client=lambda: cast(Client, evaluated)),
    )
    push_code = main(
        ["langsmith", "push", str(folder), "--data", str(golden)],
        builders=replace(UNBUILDABLE, client=lambda: cast(Client, pushed)),
    )

    assert (eval_code, push_code) == (0, 0)
    assert list(evaluated.datasets) == list(pushed.datasets) == ["shop/golden"]
    sync = LangSmithDatasetSync(
        FakeLangSmithClient(),
        namespace=namespace or core_settings.data.example_namespace,
    )
    ids = {sync.example_id("shop/golden", case_id) for case_id in ("a1", "a2")}
    assert evaluated.examples.keys() == pushed.examples.keys() == ids


def _own_error_rate(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Report an error_rate of its own beside a key summarize does not report."""
    return {
        "results": [{"key": "error_rate", "score": 0.5}, {"key": "own", "score": 1}]
    }


def colliding_graders(_config: AgentConfig) -> Graders:
    """Return Graders whose summary evaluator reports summarize's error_rate key."""
    return Graders(summary_evaluators=(_own_error_rate,))


def test_summarize_keys_win_over_a_graders_summary_key(tmp_path: Path) -> None:
    """A [graders] summary key summarize also reports is dropped; others join."""
    graders = '[graders]\nfactory = "tests.test_cli_commands:colliding_graders"\n'
    folder = agent_folder(tmp_path, graders)
    data = write_jsonl(tmp_path / "g.jsonl", [_CASE])
    results = tmp_path / "results"

    code = main(
        ["eval", str(folder), "--data", str(data), "--local"]
        + ["--results-dir", str(results)]
    )

    (run_dir,) = results.iterdir()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert (code, summary["error_rate"], summary["own"]) == (0, 0.0, 1.0)
