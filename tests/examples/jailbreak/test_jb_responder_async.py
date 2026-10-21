"""The jailbreak responder: prod's detector call, the reported turn, the replies."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from src.toolsets.jailbreak_detection.types import ConversationTurn

from examples.jailbreak import responder
from examples.jailbreak.responder import factory
from tests.connect.connect_fakes import agent_config
from tests.examples.jailbreak.jb_fakes import BLOCK, ESCALATE, FakeDetector
from tests.fakes import FakeChatModel, scripted
from trajectory_evals.connect.bridge import (
    AgentRequest,
    Responder,
    agent_card,
    build_app,
)
from trajectory_evals.connect.client import A2AAgent
from trajectory_evals.connect.settings import settings as connect_settings
from trajectory_evals.core.agent_config import AgentConfig, BridgeSection
from trajectory_evals.core.types import AgentError, ChatCompletionMessage

QUESTION = "Is the red one available?"
SYSTEM_PROMPT = "You sell cars."
DEFLECTION = "I can only help with our cars."
EARLIER: tuple[ChatCompletionMessage, ...] = (
    {"role": "user", "content": "Hi, I want a sedan."},
    {"role": "assistant", "content": "Happy to help!"},
)
EARLIER_TURNS = [
    ConversationTurn(initiator="Customer", content="Hi, I want a sedan."),
    ConversationTurn(initiator="Agent", content="Happy to help!"),
]
ASKED = ConversationTurn(initiator="Customer", content=QUESTION)
# Only a replying responder reads files, so the detector's folder needs none.
CONFIG = agent_config()


@pytest.fixture
def assistant_folder(tmp_path: Path) -> AgentConfig:
    """Return an agent folder whose [bridge] options reply, with both prompt files."""
    (tmp_path / "system.md").write_text(f"{SYSTEM_PROMPT}\n")
    (tmp_path / "deflection.md").write_text(f"{DEFLECTION}\n")
    options = {
        "reply": True,
        "model": "fake:sales",
        "system_prompt": "system.md",
        "deflection": "deflection.md",
    }
    bridge = BridgeSection(
        responder="examples.jailbreak.responder:factory", options=options
    )
    return replace(agent_config(), root=tmp_path, bridge=bridge)


def _request(
    message: str = QUESTION,
    history: tuple[ChatCompletionMessage, ...] = (),
    **fields: Any,
) -> AgentRequest:
    """Return a request for message after history, on sms unless context says."""
    defaults: dict[str, Any] = {
        "context_id": "ctx-1",
        "prompt_overrides": {},
        "context": {"channel": "sms"},
    }
    return AgentRequest(message=message, history=history, **{**defaults, **fields})


def _assistant(
    folder: AgentConfig, detector: FakeDetector, model: FakeChatModel
) -> Responder:
    """Return the folder's replying responder over detector, answering with model."""
    assert folder.bridge is not None
    return factory(
        folder.bridge.options, folder, detector=detector, model_factory=lambda _: model
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_", "call"),
    [
        (_request(), (QUESTION, (ASKED,))),
        (_request(context={"channel": "email"}), (None, (ASKED,))),
        (_request(context={"channel": "submission_form"}), (QUESTION, ())),
        (_request(history=EARLIER), (QUESTION, (*EARLIER_TURNS, ASKED))),
        (
            _request(history=EARLIER, context={"channel": "submission_form"}),
            (QUESTION, (*EARLIER_TURNS, ASKED)),
        ),
        (_request(context={}), (QUESTION, (ASKED,))),
    ],
    ids=[
        "sms",
        "email",
        "submission_form_opener",
        "history",
        "submission_form_later_turn",
        "no_channel",
    ],
)
async def test_the_detector_is_called_as_prod_calls_it(
    request_: AgentRequest, call: tuple[str | None, tuple[ConversationTurn, ...]]
) -> None:
    """Email sends only the conversation, a form opener only its text."""
    detector = FakeDetector()

    await factory({}, CONFIG, detector=detector)(request_)

    assert detector.checks == [call]


@pytest.mark.asyncio
async def test_history_keeps_initiators_and_drops_tool_messages() -> None:
    """A named agent turn keeps its initiator; earlier tool calls are left out."""
    detector = FakeDetector()
    call = {"id": "call_1", "type": "function", "function": {"name": "x"}}
    tool_result: Any = {"role": "tool", "content": "{}", "tool_call_id": "call_1"}
    named: Any = {"role": "assistant", "name": "Salesperson", "content": "Hello!"}
    history: tuple[ChatCompletionMessage, ...] = (
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        tool_result,
        named,
    )

    await factory({}, CONFIG, detector=detector)(_request(history=history))

    [(_, turns)] = detector.checks
    assert [(turn.initiator, turn.content) for turn in turns] == [
        ("Customer", "Hi"),
        ("Salesperson", "Hello!"),
        ("Customer", QUESTION),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "verdict", "result"),
    [
        (QUESTION, "ALLOWED", (False, "screen", "none", False, False, False)),
        (f"{BLOCK} now", "BLOCKED", (True, "judge", "guardrail", False, False, True)),
        (f"{ESCALATE} now", "ALLOWED", (False, "judge", "screen", False, True, False)),
        (
            "FAILOPEN now",
            "ALLOWED",
            (False, "fail_policy", "screen", True, False, False),
        ),
    ],
)
async def test_a_detector_turn_reports_its_result_route_and_verdict(
    message: str, verdict: str, result: tuple[bool, str, str, bool, bool, bool]
) -> None:
    """The reply is the verdict, after the tool result with its call log."""
    reply = await factory({}, CONFIG, detector=FakeDetector())(
        _request(message, history=EARLIER)
    )

    call, tool_result, final = reply.messages
    [tool_call] = call["tool_calls"] or []
    keys = (
        "is_jailbreak",
        "decided_by",
        "escalated_by",
        "failed_open",
        "history_read",
        "judge_reminded",
    )
    assert (tool_call["id"], tool_call["function"]["name"]) == (
        "call_2",
        "detect_jailbreak",
    )
    assert tool_result == {
        "role": "tool",
        "content": json.dumps(dict(zip(keys, result, strict=True))),
        "tool_call_id": "call_2",
    }
    assert final == {"role": "assistant", "content": verdict}
    assert (reply.text, reply.prompt_overrides_applied) == (verdict, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("channel", "earlier", "text", "history"),
    [
        ("sms", EARLIER, QUESTION, [*EARLIER_TURNS, ASKED]),
        ("email", EARLIER, None, [*EARLIER_TURNS, ASKED]),
        ("submission_form", (), QUESTION, []),
    ],
)
async def test_the_reported_call_holds_what_the_detector_got(
    channel: str,
    earlier: tuple[ChatCompletionMessage, ...],
    text: str | None,
    history: list[ConversationTurn],
) -> None:
    """The call's arguments are the routed text and history, not the raw message."""
    detector = FakeDetector()

    reply = await factory({}, CONFIG, detector=detector)(
        _request(history=earlier, context={"channel": channel})
    )

    [tool_call] = reply.messages[0]["tool_calls"] or []
    turns = [{"initiator": turn.initiator, "content": turn.content} for turn in history]
    assert json.loads(tool_call["function"]["arguments"]) == {
        "text": text,
        "history": turns,
    }
    assert detector.checks == [(text, tuple(history))]


@pytest.mark.asyncio
async def test_concurrent_turns_each_report_their_own_route() -> None:
    """Messages screened at once never see each other's call log."""
    respond = factory({}, CONFIG, detector=FakeDetector())

    replies = await asyncio.gather(
        respond(_request(f"{BLOCK} now")), respond(_request())
    )

    routes = [json.loads(str(reply.messages[1]["content"])) for reply in replies]
    assert [route["escalated_by"] for route in routes] == ["guardrail", "none"]


@pytest.mark.asyncio
async def test_the_assistant_replies_to_an_allowed_message_under_its_prompt(
    assistant_folder: AgentConfig,
) -> None:
    """An allowed message gets the model's reply to the system prompt and chat."""
    model = scripted("It is, come by anytime!")

    reply = await _assistant(assistant_folder, FakeDetector(), model)(
        _request(history=EARLIER)
    )

    assert reply.text == "It is, come by anytime!"
    [prompt] = model.prompts
    assert [(message.type, message.content) for message in prompt] == [
        ("system", SYSTEM_PROMPT),
        ("human", "Hi, I want a sedan."),
        ("ai", "Happy to help!"),
        ("human", QUESTION),
    ]


@pytest.mark.asyncio
async def test_the_assistant_deflects_a_blocked_message_without_its_model(
    assistant_folder: AgentConfig,
) -> None:
    """A blocked message gets the deflection, and the model is never called."""
    model = scripted()

    reply = await _assistant(assistant_folder, FakeDetector(), model)(
        _request(f"{BLOCK} now")
    )

    assert (reply.text, model.prompts) == (DEFLECTION, [])


@pytest.mark.asyncio
async def test_the_assistant_applies_a_system_prompt_override(
    assistant_folder: AgentConfig,
) -> None:
    """prompt_overrides["system"] replaces the system prompt and is acknowledged."""
    model = scripted("Sure.")
    request = _request(prompt_overrides={"system": "Be terse."})

    reply = await _assistant(assistant_folder, FakeDetector(), model)(request)

    assert reply.prompt_overrides_applied
    assert model.prompts[0][0].content == "Be terse."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("replies", "overrides"),
    [(False, {"system": "Be terse."}), (True, {"greeting": "Hi"})],
    ids=["detector_mode", "unknown_prompt"],
)
async def test_overrides_it_cannot_apply_are_refused_before_screening(
    assistant_folder: AgentConfig, replies: bool, overrides: dict[str, str]
) -> None:
    """The reply is empty and unacknowledged, and the detector never runs."""
    detector = FakeDetector()
    assert assistant_folder.bridge is not None
    options = assistant_folder.bridge.options if replies else {}
    respond = factory(
        options, assistant_folder, detector=detector, model_factory=lambda _: scripted()
    )

    reply = await respond(_request(prompt_overrides=overrides))

    assert (reply.text, reply.prompt_overrides_applied) == ("", False)
    assert detector.checks == []


@pytest.mark.asyncio
async def test_a_conversation_over_a2a_screens_each_turn_with_its_history(
    assistant_folder: AgentConfig,
) -> None:
    """The second turn's detector call sees the first turn's chat, not its tools."""
    detector = FakeDetector()
    model = scripted("We have two.", "Sunday 10 to 4.")
    card = agent_card(agent_config(), connect_settings.bridge.url)
    app = build_app(_assistant(assistant_folder, detector, model), card)

    async with A2AAgent.in_process(app) as agent:
        first = await agent.send("Any trucks?", context={"channel": "sms"})
        second = await agent.send(
            "Hours on Sunday?", context_id=first.context_id, context={"channel": "sms"}
        )

    assert (first.text, second.text) == ("We have two.", "Sunday 10 to 4.")
    roles = [message["role"] for message in second.messages]
    assert roles == ["assistant", "tool", "assistant"]
    assert [(turn.initiator, turn.content) for turn in detector.checks[1][1]] == [
        ("Customer", "Any trucks?"),
        ("Agent", "We have two."),
        ("Customer", "Hours on Sunday?"),
    ]


def test_replying_needs_both_prompt_files() -> None:
    """reply = true without a system prompt or deflection is rejected."""
    with pytest.raises(ValidationError, match="system_prompt and deflection"):
        factory({"reply": True}, CONFIG, detector=FakeDetector())


def test_the_real_detector_refuses_to_start_without_aws_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without credentials it would fail open, so building it raises instead."""
    monkeypatch.setattr(responder, "_has_aws_credentials", lambda: False)

    with pytest.raises(AgentError, match="aws sso login"):
        factory({}, CONFIG)
