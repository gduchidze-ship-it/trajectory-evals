"""Tests for the check and serve commands."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette

from tests.e2e.e2e_agent import (
    AGENT_TOML,
    MODELS_TOML,
    RECEIVED,
    REPLY,
    UNBUILDABLE,
    agent_folder,
    one_error_line,
    write_jsonl,
)
from trajectory_evals.cli import main
from trajectory_evals.connect.client import A2AAgent

_CASE = {"id": "a1", "input": "Where is A1?"}
_SCENARIO = {"id": "s1", "persona": "benign_curious", "scenario": "Ask about A1."}
_OPTIMIZE = (
    '[optimize]\nprompts = { system = "policy.md" }\n'
    "floors = { pass_rate = 0.8, max_error_rate = 0.1 }\n"
)


@pytest.fixture(autouse=True)
def received() -> Iterator[list[str]]:
    """Clear the test agent's record of messages before and after each test."""
    RECEIVED.clear()
    yield RECEIVED
    RECEIVED.clear()


def test_check_validates_tables_datasets_and_card_offline(
    tmp_path: Path,
    network_attempts: list[Any],
    received: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """check counts each dataset by its kind, resolves the card and sends nothing."""
    folder = agent_folder(tmp_path, MODELS_TOML + _OPTIMIZE)
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
    ("tables", "scenario", "message"),
    [
        ('[simulation]\nuser_model = "gpt-5"\n', None, "not a provider:model"),
        ('[simulation]\nstop_when = "nowhere.at_all:x"\n', None, "nowhere"),
        ('[judges.tone]\nprompt = "missing.md"\n', None, "file not found"),
        ("", {"id": "s1", "persona": "benign_curious"}, "jsonl:1: scenario: Field"),
        ("", {**_SCENARIO, "persona": "ghost"}, "unknown personas ['ghost']"),
        (MODELS_TOML + 'personas = ["ghost"]\n', None, "must be attack personas"),
        (
            MODELS_TOML + '[optimize]\nprompts = { system = "policy.md" }\n',
            None,
            "floors",
        ),
        (_OPTIMIZE, None, "redteam is on"),
        (
            MODELS_TOML
            + _OPTIMIZE
            + '[variants.short]\nprompt_overrides = { tone = "policy.md" }\n'
            + "cost = 0.5\n",
            None,
            "overrides prompts ['tone']",
        ),
        ('[langsmith]\nprojet = "dev-sales"\n', None, "projet"),
    ],
)
def test_check_rejects_what_a_run_command_would(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tables: str,
    scenario: dict[str, str] | None,
    message: str,
) -> None:
    """A table or dataset a run command would reject makes check exit 2.

    That covers a model, hook, judge, scenario, persona, [optimize], [variants]
    and [langsmith].
    """
    folder = agent_folder(tmp_path, tables)
    if scenario is not None:
        write_jsonl(folder / "datasets" / "scenarios.jsonl", [scenario])

    code = main(["check", str(folder)])

    assert message in one_error_line(code, capsys)


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

    assert "agent card" in one_error_line(code, capsys)
    assert network_attempts


def test_check_reports_a_responder_that_does_not_import(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A [bridge] responder module that does not exist exits 2 with its name."""
    folder = agent_folder(tmp_path, "")
    toml = AGENT_TOML.replace("tests.e2e.e2e_agent:factory", "no_such_module:factory")
    (folder / "agent.toml").write_text(toml, encoding="utf-8")

    code = main(["check", str(folder)])

    assert "no_such_module" in one_error_line(code, capsys)


class _Server:
    """Records what serve was asked to serve instead of binding a socket."""

    def __init__(self) -> None:
        """Start with nothing served."""
        self.calls: list[tuple[Starlette, str, int]] = []

    def __call__(self, app: Starlette, host: str, port: int) -> None:
        """Record the app, host and port."""
        self.calls.append((app, host, port))


@pytest.mark.parametrize(
    ("host", "url", "warnings"),
    [
        ("127.0.0.1", "http://127.0.0.1:8001/", []),
        ("localhost", "http://localhost:8001/", []),
        ("::1", "http://[::1]:8001/", []),
        ("10.0.0.5", "http://10.0.0.5:8001/", ["no inbound auth"]),
        ("0.0.0.0", "http://0.0.0.0:8001/", ["no inbound auth", "pass --url"]),
        ("::", "http://[::]:8001/", ["no inbound auth", "pass --url"]),
    ],
)
@pytest.mark.asyncio
async def test_serve_advertises_a_card_url_a_client_can_use(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    received: list[str],
    host: str,
    url: str,
    warnings: list[str],
) -> None:
    """The card's URL brackets IPv6, and a client sending to it reaches the agent.

    Binding off loopback warns, and so does binding every address.
    """
    server = _Server()
    argv = ["serve", str(agent_folder(tmp_path)), "--host", host, "--port", "8001"]

    code = main(argv, builders=replace(UNBUILDABLE, serve=server))

    [(app, served_host, port)] = server.calls
    assert (code, served_host, port) == (0, host, 8001)
    errors = capsys.readouterr().err
    assert [line for line in warnings if line in errors] == warnings
    assert errors.count("\n") == len(warnings)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    async with A2AAgent(http, url) as agent:
        turn = await agent.send("Where is A1?")
    assert (turn.text, received) == (REPLY, ["Where is A1?"])


@pytest.mark.asyncio
async def test_serve_advertises_url_when_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], received: list[str]
) -> None:
    """--url replaces the advertised address, so binding everywhere does not warn."""
    server = _Server()
    url = "http://agent.example:8001/"
    folder = str(agent_folder(tmp_path))

    code = main(
        ["serve", folder, "--host", "0.0.0.0", "--port", "8001", "--url", url],
        builders=replace(UNBUILDABLE, serve=server),
    )

    [(app, _, _)] = server.calls
    assert code == 0
    assert "pass --url" not in capsys.readouterr().err
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    async with A2AAgent(http, url) as agent:
        turn = await agent.send("Hi")
    assert turn.text == REPLY


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

    code = main(["serve", str(folder)], builders=UNBUILDABLE)

    assert "has no [bridge] table" in one_error_line(code, capsys)
