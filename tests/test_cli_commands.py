"""Tests for the CLI's parsing, preflight, bad-input exits, check, serve and uploads."""

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
from starlette.applications import Starlette

from tests.e2e.e2e_agent import (
    AGENT_TOML,
    RECEIVED,
    REPLY,
    agent_folder,
    write_jsonl,
)
from trajectory_evals.cli import DEFAULT_BUILDERS, Builders, build_parser, main
from trajectory_evals.core import runner

_CASE = {"id": "a1", "input": "Where is A1?"}
_SCENARIO = {"id": "s1", "persona": "benign_curious", "scenario": "Ask about A1."}
_EXPERIMENT = "uploaded-1"


def _refuse(*_: object) -> NoReturn:
    """Fail the test, because nothing may be built before the input is checked."""
    raise AssertionError("a collaborator was built before the input was checked")


_UNBUILDABLE = Builders(
    client=_refuse,
    local_client=_refuse,
    agent=_refuse,
    model_factory=_refuse,
    serve=_refuse,
)


@pytest.fixture(autouse=True)
def received() -> Iterator[list[str]]:
    """Clear the test agent's record of messages before and after each test."""
    RECEIVED.clear()
    yield RECEIVED
    RECEIVED.clear()


def test_eval_defaults() -> None:
    """eval runs uploaded, 4 at once, once each, segmented by risk and difficulty."""
    args = build_parser().parse_args(["eval", "shop", "--data", "g.jsonl"])

    assert (args.agent, args.data) == ("shop", Path("g.jsonl"))
    assert (args.local, args.limit, args.prefix) == (False, None, None)
    assert (args.concurrency, args.repetitions) == (4, 1)
    assert args.results_dir == Path("results")
    assert args.segment_by == ("risk", "difficulty")


def test_redteam_defaults_defer_to_agent_toml() -> None:
    """redteam reads no data file and leaves personas and attempts to [redteam]."""
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
    "argv", [["eval", "shop"], ["simulate", "shop"], ["optimize", "shop"]]
)
def test_usage_errors_exit_2(argv: list[str]) -> None:
    """A missing --data or --scenarios, or an unknown command, is a usage error."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(argv)

    assert exit_info.value.code == 2


def _one_error_line(code: int, capsys: pytest.CaptureFixture[str]) -> str:
    """Assert main exited 2 with one stderr line, and return that line."""
    error = capsys.readouterr().err
    assert (code, error.count("\n")) == (2, 1)
    assert error.startswith("python -m trajectory_evals: error: ")
    return error


def test_uploaded_run_without_a_langsmith_key_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --local, an unset LANGSMITH_API_KEY stops the run before any build."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    data = write_jsonl(tmp_path / "g.jsonl", [_CASE])

    code = main(["eval", "shop", "--data", str(data)], builders=_UNBUILDABLE)

    assert "LANGSMITH_API_KEY is not set" in _one_error_line(code, capsys)


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "shop", "--data", "missing.jsonl", "--local"],
        ["simulate", "shop", "--scenarios", "missing.jsonl", "--local"],
    ],
)
def test_missing_data_file_exits_2_before_any_build(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """A data file that does not exist stops the run before the agent is loaded."""
    code = main(argv, builders=_UNBUILDABLE)

    assert "data file not found: missing.jsonl" in _one_error_line(code, capsys)


def test_unknown_agent_folder_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An agent path with no agent.toml is reported, not raised."""
    code = main(["redteam", str(tmp_path / "nobody"), "--local"], builders=_UNBUILDABLE)

    assert "file not found" in _one_error_line(code, capsys)


def test_bad_row_exits_2_without_echoing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad golden row names path:line and its field, never its text."""
    folder = agent_folder(tmp_path)
    secret = "my card number is 4111"
    data = write_jsonl(tmp_path / "g.jsonl", [_CASE, {"id": "b", "inputs": secret}])

    argv = ["eval", str(folder), "--data", str(data), "--local"]

    code = main(argv, builders=_UNBUILDABLE)

    error = _one_error_line(code, capsys)
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

    code = main(argv, builders=_UNBUILDABLE)

    assert f"no cases to run in {data}" in _one_error_line(code, capsys)


def test_unknown_redteam_persona_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A persona name that is not an attack persona stops the run before a build."""
    folder = agent_folder(tmp_path)

    code = main(
        ["redteam", str(folder), "--personas", "benign_curious", "--local"],
        builders=_UNBUILDABLE,
    )

    assert "must be attack personas" in _one_error_line(code, capsys)


def test_check_validates_tables_datasets_and_card_offline(
    tmp_path: Path,
    network_attempts: list[Any],
    received: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """check counts each dataset by its kind, resolves the card and sends nothing."""
    folder = agent_folder(tmp_path)
    write_jsonl(folder / "datasets" / "golden.sample.jsonl", [_CASE])
    write_jsonl(folder / "datasets" / "scenarios.sample.jsonl", [_SCENARIO])

    code = main(["check", str(folder)])

    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        f"agent: shop ({folder.resolve()})",
        "datasets/golden.sample.jsonl: 1 cases",
        "datasets/scenarios.sample.jsonl: 1 cases",
        "tables: ok",
        "card: ok",
    ]
    assert received == []
    assert network_attempts == []


def test_check_ping_sends_one_message(
    tmp_path: Path, received: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """--ping sends one message and prints the agent's reply."""
    code = main(["check", str(agent_folder(tmp_path)), "--ping"])

    assert code == 0
    assert received == ["Hello"]
    assert capsys.readouterr().out.splitlines()[-1] == f"ping: {REPLY}"


@pytest.mark.parametrize(
    ("tables", "dataset", "message"),
    [
        ('[simulation]\nuser_model = "gpt-5"\n', None, "not a provider:model"),
        ('[simulation]\nstop_when = "nowhere.at_all:x"\n', None, "nowhere"),
        ('[judges.tone]\nprompt = "missing.md"\n', None, "file not found"),
        ("", {"id": "s1", "persona": "benign_curious"}, "jsonl:1: scenario: Field"),
    ],
)
def test_check_reports_a_bad_table_or_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tables: str,
    dataset: dict[str, str] | None,
    message: str,
) -> None:
    """A bad model name, hook path, judge prompt or scenario row exits 2."""
    folder = agent_folder(tmp_path, tables)
    if dataset is not None:
        write_jsonl(folder / "datasets" / "scenarios.jsonl", [dataset])

    code = main(["check", str(folder)])

    assert message in _one_error_line(code, capsys)


def test_check_reports_an_unreachable_card(
    tmp_path: Path, network_attempts: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """An agent url that refuses connections exits 2 instead of raising."""
    folder = tmp_path / "remote"
    folder.mkdir()
    (folder / "agent.toml").write_text(
        '[agent]\nurl = "http://agent.test/"\ndescription = "Remote."\n',
        encoding="utf-8",
    )

    code = main(["check", str(folder)])

    assert "agent card" in _one_error_line(code, capsys)
    assert network_attempts


def test_check_reports_a_responder_that_does_not_import(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A [bridge] responder module that does not exist exits 2 with its name."""
    folder = agent_folder(tmp_path, "")
    toml = AGENT_TOML.replace("tests.e2e.e2e_agent:factory", "no_such_module:factory")
    (folder / "agent.toml").write_text(toml, encoding="utf-8")

    code = main(["check", str(folder)])

    assert "no_such_module" in _one_error_line(code, capsys)


class _Server:
    """Records what serve was asked to serve instead of binding a socket."""

    def __init__(self) -> None:
        """Start with nothing served."""
        self.calls: list[tuple[Starlette, str, int]] = []

    def __call__(self, app: Starlette, host: str, port: int) -> None:
        """Record the app, host and port."""
        self.calls.append((app, host, port))


@pytest.mark.parametrize(
    ("host", "warns"),
    [("127.0.0.1", False), ("localhost", False), ("::1", False), ("0.0.0.0", True)],
)
def test_serve_warns_only_off_loopback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], host: str, warns: bool
) -> None:
    """serve hands the bridge app to the server, warning when it binds publicly."""
    server = _Server()
    folder = agent_folder(tmp_path)

    code = main(
        ["serve", str(folder), "--host", host, "--port", "8001"],
        builders=replace(_UNBUILDABLE, serve=server),
    )

    [(app, served_host, port)] = server.calls
    assert (code, served_host, port) == (0, host, 8001)
    assert isinstance(app, Starlette)
    assert ("no inbound auth" in capsys.readouterr().err) is warns


def test_serve_without_a_bridge_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An agent reached by url has nothing to serve."""
    folder = tmp_path / "remote"
    folder.mkdir()
    (folder / "agent.toml").write_text(
        '[agent]\nurl = "http://agent.test"\ndescription = "Remote."\n',
        encoding="utf-8",
    )

    code = main(["serve", str(folder)], builders=_UNBUILDABLE)

    assert "has no [bridge] table" in _one_error_line(code, capsys)


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
    ("tables", "dataset"),
    [
        ("", "shop/golden"),
        ('[langsmith]\ndataset_prefix = "support"\n', "support/golden"),
    ],
)
def test_uploaded_eval_syncs_the_named_dataset_on_the_uploading_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tables: str,
    dataset: str,
) -> None:
    """Without --local the cases sync to <prefix>/<stem> and the run uploads."""
    calls: list[dict[str, Any]] = []

    async def aevaluate(target: Any, /, **kwargs: Any) -> _FakeResults:
        """Record the call and return results with no rows."""
        calls.append(kwargs)
        return _FakeResults()

    monkeypatch.setattr(runner, "aevaluate", aevaluate)
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-fake")
    client = _RecordingClient()
    folder = agent_folder(tmp_path, tables)
    data = write_jsonl(tmp_path / "golden.jsonl", [_CASE])
    builders = replace(DEFAULT_BUILDERS, client=lambda: cast(Client, client))

    code = main(
        ["eval", str(folder), "--data", str(data), "--results-dir", str(tmp_path)],
        builders=builders,
    )

    [call] = calls
    assert (code, client.created) == (0, [dataset])
    assert (call["data"], call["client"], call["upload_results"]) == (
        dataset,
        client,
        True,
    )
    assert call["experiment_prefix"] == "shop-eval"
    assert call["metadata"] == {
        "agent": "shop",
        "command": "eval",
        "dataset_file": "golden.jsonl",
        "history": "metadata",
    }
    assert f"experiment: {_EXPERIMENT}" in capsys.readouterr().out
