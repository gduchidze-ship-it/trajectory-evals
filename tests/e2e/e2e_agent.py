"""A bridged test agent that calls one tool per turn, its folders, CLI test helpers."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

import pytest

from trajectory_evals.cli import INPUT_ERROR_EXIT_CODE, Builders
from trajectory_evals.connect.bridge import AgentReply, AgentRequest, Responder
from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.types import ChatCompletionMessage, ToolMessage

TOOL = "lookup_order"
REPLY = "Order A1 ships tomorrow."
DESCRIPTION = "A support agent for an online bike shop."
RESPONDER = "tests.e2e.e2e_agent:factory"
POLICY = "Never promise a refund."
# Every message the bridged agent received, in order; tests clear it.
RECEIVED: list[str] = []

AGENT_TOML = f"""
[agent]
description = "{DESCRIPTION}"

[bridge]
responder = "{RESPONDER}"
options = {{ reply = "{REPLY}" }}
"""
# Model names the tests' model factories map to scripted models.
MODELS_TOML = """
[simulation]
user_model = "user:scripted"

[redteam]
policy = "policy.md"
judge_model = "judge:scripted"
"""


def _turn(text: str) -> tuple[ChatCompletionMessage, ...]:
    """Return one turn: a lookup_order call, its result, then the reply."""
    call_id = f"call-{len(RECEIVED)}"
    function = {"name": TOOL, "arguments": json.dumps({"order_id": "A1"})}
    call = {"id": call_id, "type": "function", "function": function}
    result: ToolMessage = {
        "role": "tool",
        "content": "shipped",
        "tool_call_id": call_id,
    }
    return (
        {"role": "assistant", "content": "", "tool_calls": [call]},
        result,
        {"role": "assistant", "content": text},
    )


def factory(options: Mapping[str, Any], _config: AgentConfig) -> Responder:
    """Return a responder that records each message and answers options' reply."""
    reply = str(options["reply"])

    async def respond(request: AgentRequest) -> AgentReply:
        """Record the message, then call the tool and reply."""
        RECEIVED.append(request.message)
        return AgentReply(text=reply, messages=_turn(reply))

    return respond


def agent_folder(tmp_path: Path, tables: str = MODELS_TOML) -> Path:
    """Write an agent folder serving the test agent, with tables and its policy."""
    root = tmp_path / "shop"
    root.mkdir()
    (root / "agent.toml").write_text(AGENT_TOML + tables, encoding="utf-8")
    (root / "policy.md").write_text(POLICY, encoding="utf-8")
    return root


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write rows as JSON lines to path, creating its folder, and return path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def refuse(*_: object) -> NoReturn:
    """Fail the test, because nothing may be built before the input is checked."""
    raise AssertionError("a collaborator was built before the input was checked")


UNBUILDABLE = Builders(
    client=refuse,
    local_client=refuse,
    agent=refuse,
    model_factory=refuse,
    serve=refuse,
)


def one_error_line(code: int, capsys: pytest.CaptureFixture[str]) -> str:
    """Assert main exited for bad input with one stderr line; return that line."""
    error = capsys.readouterr().err
    assert (code, error.count("\n")) == (INPUT_ERROR_EXIT_CODE, 1)
    assert error.startswith("python -m trajectory_evals: error: ")
    return error
