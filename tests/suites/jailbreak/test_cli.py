"""Tests for the command line's parsing, preflight checks and upload wiring."""

from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from botocore.exceptions import ProfileNotFound, UnauthorizedSSOTokenError
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import EvaluationResult
from langsmith.utils import LangSmithNotFoundError

from trajectory_evals.suites.jailbreak import cli, runner
from trajectory_evals.suites.jailbreak.cli import (
    DEFAULT_BUILDERS,
    Builders,
    Preflight,
    _required_env,
    _simulation_extra,
    build_parser,
    main,
    run_detection,
    run_simulation,
)
from trajectory_evals.suites.jailbreak.settings import settings as eval_settings
from trajectory_evals.suites.jailbreak.simulation import PERSONAS
from src.toolsets.jailbreak_detection.service import JailbreakDetectionService
from src.toolsets.jailbreak_detection.settings import settings
from tests.suites.jailbreak.cli_fakes import case_row, write_jsonl
from tests.suites.jailbreak.simulation.simulation_fakes import (
    FakeChatModel,
    FakeDetector,
    scripted,
)

_KEYS = {"LANGSMITH_API_KEY": "lsv2-fake", "OPENAI_API_KEY": "sk-fake"}
_EXPERIMENT = "uploaded-1"


def _refuse(*_: object) -> NoReturn:
    """Fail the test, because nothing may be built before preflight passes."""
    raise AssertionError("a collaborator was built before preflight passed")


_UNBUILDABLE = Builders(
    client=_refuse, local_client=_refuse, detector=_refuse, chat_model=_refuse
)


def _use_data_dir(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    """Point the command line's data directory at data_dir for one test."""
    paths = eval_settings.paths.model_copy(update={"data_dir": str(data_dir)})
    moved = eval_settings.model_copy(update={"paths": paths})
    monkeypatch.setattr(cli, "settings", moved)


def _passing_preflight() -> Preflight:
    """Return a preflight with every key set and AWS credentials resolving."""
    return Preflight(environ=_KEYS, has_aws_credentials=lambda: True)


class _RecordingClient:
    """A LangSmith client stand-in whose datasets start empty and record their names."""

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
    """The parts of aevaluate's results that the runner reads."""

    experiment_name = _EXPERIMENT

    def __init__(self) -> None:
        """Hold one summary score where aevaluate keeps it."""
        self._summary_results = {"results": [EvaluationResult(key="score", score=1)]}

    async def get_comparison_url(self) -> str:
        """Return the experiment's URL."""
        return "https://smith.example/compare"

    def __len__(self) -> int:
        """Return the number of runs."""
        return 1


class _RecordingAevaluate:
    """An aevaluate stand-in that records the keyword arguments of each call."""

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, target: Any, /, **kwargs: Any) -> _FakeResults:
        """Record the call and return fixed results."""
        self.calls.append(kwargs)
        return _FakeResults()


@pytest.fixture
def aevaluate(monkeypatch: pytest.MonkeyPatch) -> _RecordingAevaluate:
    """Replace the runner's aevaluate with a recorder."""
    recorder = _RecordingAevaluate()
    monkeypatch.setattr(runner, "aevaluate", recorder)
    return recorder


def test_detection_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Detection runs every data/jailbreak JSONL, sorted, with the spec's defaults."""
    names = ["b.jsonl", "a.jsonl", "notes.txt"]
    for name in names:
        (tmp_path / name).touch()
    _use_data_dir(monkeypatch, tmp_path)

    args = build_parser().parse_args(["detection"])

    assert args.data == [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    assert (args.limit, args.max_concurrency, args.repetitions) == (None, 8, 1)
    assert (args.local, args.experiment_prefix) == (False, "detect-jailbreak")
    assert (args.run, args.needs_openai) == (run_detection, False)


def test_simulation_defaults() -> None:
    """Simulation samples the prod benign file with every persona and spec defaults."""
    args = build_parser().parse_args(["simulation"])

    assert args.data == [eval_settings.paths.data_path / "langsmith_prod_benign.jsonl"]
    assert (args.sample, args.seed, args.max_turns) == (20, 0, 6)
    assert args.personas == list(PERSONAS)
    assert (args.user_model, args.responder_model, args.judge_model) == (
        "gpt-5.4-mini",
        "gpt-5.4-mini",
        "gpt-5.5",
    )
    assert (args.max_concurrency, args.repetitions) == (4, 1)
    assert (args.local, args.experiment_prefix) == (False, "simulate-jailbreak")
    assert (args.run, args.needs_openai) == (run_simulation, True)


def test_simulation_has_no_limit_option(capsys: pytest.CaptureFixture[str]) -> None:
    """Only detection takes --limit; the simulation sample is sized by --sample."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["simulation", "--limit", "2"])

    assert exit_info.value.code == 2
    assert "unrecognized arguments: --limit 2" in capsys.readouterr().err


def test_repeated_personas_run_once_each_in_first_order() -> None:
    """A persona named twice is kept once, where it first appeared."""
    argv = ["simulation", "--personas", "roleplay_framer", "benign_follow_up"]
    args = build_parser().parse_args([*argv, "roleplay_framer"])

    assert args.personas == ["roleplay_framer", "benign_follow_up"]


def test_unknown_persona_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    """A persona name outside the registry is a usage error that exits 2."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["simulation", "--personas", "nobody"])

    assert exit_info.value.code == 2
    assert "invalid choice: 'nobody'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "option"),
    [
        ("detection", "--limit"),
        ("detection", "--max-concurrency"),
        ("simulation", "--max-concurrency"),
        ("simulation", "--repetitions"),
        ("simulation", "--sample"),
        ("simulation", "--max-turns"),
    ],
)
@pytest.mark.parametrize("value", ["0", "-3", "two"])
def test_counts_below_one_are_rejected(
    command: str, option: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every count option refuses anything but a whole number of at least one."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([command, f"{option}={value}"])

    assert exit_info.value.code == 2
    assert "expected a whole number >= 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "environ", "aws", "message"),
    [
        (["detection"], _KEYS, False, "aws sso login --profile sales-ai-test"),
        (["detection"], {}, True, "LANGSMITH_API_KEY is not set"),
        (["simulation"], {"LANGSMITH_API_KEY": "k"}, True, "OPENAI_API_KEY is not set"),
        (["detection", "--data", "missing.jsonl"], _KEYS, True, "not found: missing"),
    ],
)
def test_preflight_failure_exits_2_before_building_anything(
    argv: list[str],
    environ: dict[str, str],
    aws: bool,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each missing prerequisite prints one error line and exits 2 before any call."""
    preflight = Preflight(environ=environ, has_aws_credentials=lambda: aws)

    code = main(argv, preflight=preflight, builders=_UNBUILDABLE)

    error = capsys.readouterr().err
    assert (code, error.count("\n")) == (2, 1)
    assert message in error


def test_preflight_fails_when_there_is_no_data_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Detection with an empty data directory exits 2 instead of running nothing."""
    _use_data_dir(monkeypatch, tmp_path)

    code = main(["detection"], preflight=_passing_preflight(), builders=_UNBUILDABLE)

    assert code == 2
    assert "no data files" in capsys.readouterr().err


def test_bad_later_file_exits_2_before_any_file_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad row in the second file stops the run before a client or detector exists."""
    secret = "my card number is 4111"
    bad_row = {"id": "b", "input_text": secret, "expected_action": "MAYBE"}
    good = write_jsonl(tmp_path / "good.jsonl", [case_row("a", "Hi", "ALLOWED")])
    bad = write_jsonl(tmp_path / "bad.jsonl", [case_row("a", "Hi", "ALLOWED"), bad_row])

    code = main(
        ["detection", "--data", str(good), str(bad)],
        preflight=_passing_preflight(),
        builders=_UNBUILDABLE,
    )

    error = capsys.readouterr().err
    assert (code, error.count("\n")) == (2, 1)
    assert f"{bad}:2: expected_action" in error
    assert secret not in error and "MAYBE" not in error


@pytest.mark.parametrize("command", ["detection", "simulation"])
@pytest.mark.parametrize("local", [[], ["--local"]])
def test_repeated_case_id_exits_2_before_any_experiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    local: list[str],
) -> None:
    """A case id used twice stops either command, local or uploaded, before a build."""
    rows = [case_row("a", "Hi", "ALLOWED"), case_row("a", "Hey", "ALLOWED")]
    data = write_jsonl(tmp_path / "repeated.jsonl", rows)

    code = main(
        [command, "--data", str(data), *local],
        preflight=_passing_preflight(),
        builders=_UNBUILDABLE,
    )

    error = capsys.readouterr().err
    assert (code, error.count("\n")) == (2, 1)
    assert f"{data}:2: case_id 'a' appears more than once" in error


@pytest.mark.parametrize(
    ("argv", "required"),
    [
        (["detection"], ["LANGSMITH_API_KEY"]),
        (["detection", "--local"], []),
        (["simulation"], ["LANGSMITH_API_KEY", "OPENAI_API_KEY"]),
        (["simulation", "--local"], ["OPENAI_API_KEY"]),
    ],
)
def test_each_command_requires_its_own_keys(
    argv: list[str], required: list[str]
) -> None:
    """LangSmith's key is needed unless local, and OpenAI's only for simulation."""
    assert list(_required_env(build_parser().parse_args(argv))) == required


def test_local_run_needs_no_langsmith_key(tmp_path: Path) -> None:
    """With --local, preflight passes without LANGSMITH_API_KEY."""
    data = write_jsonl(tmp_path / "cases.jsonl", [case_row("a", "Hi", "ALLOWED")])
    args = build_parser().parse_args(["detection", "--data", str(data), "--local"])
    preflight = Preflight(environ={}, has_aws_credentials=lambda: True)

    preflight.check(data=args.data, required_env=_required_env(args))


class _Credentials:
    """Credentials whose freezing succeeds or raises the given error."""

    def __init__(self, error: Exception | None) -> None:
        """Hold the error that freezing raises, if any."""
        self._error = error

    def get_frozen_credentials(self) -> object:
        """Raise the held error, or return frozen credentials."""
        if self._error is not None:
            raise self._error
        return object()


def _session(credentials: _Credentials | None) -> Callable[[], SimpleNamespace]:
    """Return a boto3.Session stand-in whose sessions hold credentials."""
    return lambda: SimpleNamespace(get_credentials=lambda: credentials)


def _missing_profile() -> NoReturn:
    """Fail the way boto3.Session does when AWS_PROFILE names no profile."""
    raise ProfileNotFound(profile="sales-ai-test")


@pytest.mark.parametrize(
    ("session", "resolves"),
    [
        (_session(None), False),
        (_missing_profile, False),
        (_session(_Credentials(UnauthorizedSSOTokenError())), False),
        (_session(_Credentials(None)), True),
    ],
)
def test_aws_credentials_resolve_only_when_they_can_be_frozen(
    monkeypatch: pytest.MonkeyPatch, session: Callable[[], object], resolves: bool
) -> None:
    """No credentials, an unknown profile or an expired SSO login fail the check."""
    monkeypatch.setattr(cli.boto3, "Session", session)

    assert cli.DEFAULT_PREFLIGHT.has_aws_credentials() is resolves


def test_default_builders_build_the_concrete_collaborators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped builders make ChatOpenAI, the real cascade and a no-thread client."""
    pytest.importorskip("aioboto3")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    chat_model = DEFAULT_BUILDERS.chat_model("gpt-test")

    assert isinstance(chat_model, ChatOpenAI)
    assert chat_model.model_name == "gpt-test"
    assert isinstance(DEFAULT_BUILDERS.detector(), JailbreakDetectionService)
    assert DEFAULT_BUILDERS.client is Client
    assert DEFAULT_BUILDERS.local_client().tracing_queue is None


def test_detection_upload_names_each_dataset_and_shares_one_client(
    tmp_path: Path,
    aevaluate: _RecordingAevaluate,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each file syncs to jailbreak-detection/<stem> and runs on the same client."""
    alpha = write_jsonl(tmp_path / "alpha.jsonl", [case_row("a", "Hi", "ALLOWED")])
    beta = write_jsonl(
        tmp_path / "beta.jsonl",
        [case_row("b", "Hi", "ALLOWED"), case_row("c", "Hey", "BLOCKED")],
    )
    client = _RecordingClient()
    builders = Builders(
        client=lambda: cast(Client, client),
        local_client=_refuse,
        detector=FakeDetector,
        chat_model=_refuse,
    )

    code = main(
        ["detection", "--data", str(alpha), str(beta)],
        preflight=_passing_preflight(),
        builders=builders,
    )

    names = ["jailbreak-detection/alpha", "jailbreak-detection/beta"]
    assert code == 0
    assert client.created == names
    assert [call["data"] for call in aevaluate.calls] == names
    assert all(call["client"] is client for call in aevaluate.calls)
    assert [
        (call["metadata"]["dataset_file"], call["metadata"]["row_count"])
        for call in aevaluate.calls
    ] == [("alpha.jsonl", 1), ("beta.jsonl", 2)]
    assert {call["experiment_prefix"] for call in aevaluate.calls} == {
        "detect-jailbreak"
    }
    assert capsys.readouterr().out.count(f"experiment: {_EXPERIMENT}") == 2


def test_simulation_upload_names_the_dataset_and_records_its_settings(
    tmp_path: Path, aevaluate: _RecordingAevaluate
) -> None:
    """The sample syncs to jailbreak-simulation/<stem> with the run's settings."""
    data = write_jsonl(
        tmp_path / "openers.jsonl",
        [case_row("a", "Hi", "ALLOWED"), case_row("b", "Hey", "ALLOWED", "email")],
    )
    client = _RecordingClient()
    models: list[str] = []

    def chat_model(name: str) -> FakeChatModel:
        """Record the model name and return an unscripted fake."""
        models.append(name)
        return scripted()

    builders = Builders(
        client=lambda: cast(Client, client),
        local_client=_refuse,
        detector=FakeDetector,
        chat_model=chat_model,
    )
    personas = ["benign_follow_up", "roleplay_framer"]

    code = main(
        ["simulation", "--data", str(data), "--sample", "2", "--personas", *personas]
        + ["--user-model", "u", "--responder-model", "r", "--judge-model", "j"],
        preflight=_passing_preflight(),
        builders=builders,
    )

    expected = {
        "personas": personas,
        "sample": 2,
        "seed": 0,
        "max_turns": 6,
        "simulated_user_model": "u",
        "responder_model": "r",
        "attack_judge_model": "j",
        "judge_model": settings.models.judge,
        "dataset_file": "openers.jsonl",
        "row_count": 4,
    }
    summaries = {
        "attack_block_rate",
        "benign_false_block_rate",
        "mean_turns_to_block",
        "persona_faithful_rate",
        "error_rate",
    }
    [call] = aevaluate.calls
    assert code == 0
    assert client.created == ["jailbreak-simulation/openers"]
    assert (call["data"], call["client"]) == ("jailbreak-simulation/openers", client)
    assert sorted(models) == ["j", "r", "u"]
    assert {key: call["metadata"][key] for key in expected} == expected
    assert len(call["evaluators"]) == 4
    [summary] = call["summary_evaluators"]
    assert {result["key"] for result in summary([], [])["results"]} == summaries


@pytest.mark.parametrize("local", [[], ["--local"]])
@pytest.mark.parametrize("content", ["", "\n   \n"])
def test_detection_file_without_cases_exits_2_before_any_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
    local: list[str],
) -> None:
    """An empty or blank-only data file stops detection before any client exists."""
    data = tmp_path / "empty.jsonl"
    data.write_text(content)

    code = main(
        ["detection", "--data", str(data), *local],
        preflight=_passing_preflight(),
        builders=_UNBUILDABLE,
    )

    error = capsys.readouterr().err
    assert (code, error.count("\n")) == (2, 1)
    assert f"no cases to run in {data}" in error


@pytest.mark.parametrize("local", [[], ["--local"]])
def test_simulation_file_without_openers_exits_2_before_any_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], local: list[str]
) -> None:
    """A file with no ALLOWED single-message row yields no opener and stops the run."""
    rows = [case_row("a", "Hi", "BLOCKED"), case_row("b", "Hey", "REVIEW")]
    data = write_jsonl(tmp_path / "no_openers.jsonl", rows)

    code = main(
        ["simulation", "--data", str(data), *local],
        preflight=_passing_preflight(),
        builders=_UNBUILDABLE,
    )

    error = capsys.readouterr().err
    assert (code, error.count("\n")) == (2, 1)
    assert f"no cases to run in {data}" in error


def test_option_help_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An option's help and metavar are read from [help] when the parser is built."""
    metavars = eval_settings.help.metavars.model_copy(update={"count": "ROWS"})
    help_texts = eval_settings.help.model_copy(
        update={"limit": "rows to score", "metavars": metavars}
    )
    monkeypatch.setattr(
        cli, "settings", eval_settings.model_copy(update={"help": help_texts})
    )

    with pytest.raises(SystemExit):
        build_parser().parse_args(["detection", "--help"])

    assert "--limit ROWS          rows to score" in capsys.readouterr().out


def test_simulation_metadata_keys_come_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each simulation option is recorded under the key [cli.metadata_keys] gives it."""
    keys = eval_settings.cli.metadata_keys.model_copy(update={"user_model": "user_llm"})
    cli_settings = eval_settings.cli.model_copy(update={"metadata_keys": keys})
    monkeypatch.setattr(
        cli, "settings", eval_settings.model_copy(update={"cli": cli_settings})
    )

    extra = _simulation_extra(build_parser().parse_args(["simulation"]))

    assert extra["user_llm"] == "gpt-5.4-mini"
    assert "simulated_user_model" not in extra
