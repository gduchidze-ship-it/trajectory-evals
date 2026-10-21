"""Tests for the guarded conversation app that openevals drives."""

import asyncio
from collections.abc import Mapping

import pytest
from langchain_core.language_models.fake_chat_models import ParrotFakeChatModel

from evals.jailbreak.settings import settings as eval_settings
from evals.jailbreak.simulation import GuardedConversationApp
from evals.jailbreak.types import ChatCompletionMessage
from src.toolsets.jailbreak_detection.service import JailbreakDetectionService
from src.toolsets.jailbreak_detection.settings import settings
from src.toolsets.jailbreak_detection.types import (
    ConversationTurn,
    Finding,
    InboundMessage,
    Verdict,
)
from tests.evals.jailbreak.simulation.simulation_fakes import (
    BLOCK_MARKER,
    FAKE_ESCALATED_BY,
    FakeDetector,
    scripted,
)

CUSTOMER = settings.input.customer_initiator
THREAD = "thread-1"


def _customer(content: str | list[dict[str, str]]) -> ChatCompletionMessage:
    """Return a simulated customer message as openevals passes it to the app."""
    return {"role": "user", "content": content}


def _turn(initiator: str, content: str) -> ConversationTurn:
    """Return a runner-shaped conversation turn."""
    return ConversationTurn(initiator=initiator, content=content)


async def _unreachable_screen(message: InboundMessage) -> Finding:
    """Fail the test if the service runs a detector model on the message."""
    raise AssertionError(f"screen called for {message.text!r}")


async def _unreachable_judge(
    message: InboundMessage, findings: Mapping[str, Finding]
) -> Verdict | None:
    """Fail the test if the service asks the judge about the message."""
    raise AssertionError(f"judge called for {message.text!r}")


def test_respond_passes_openevals_coroutine_check() -> None:
    """openevals awaits the app only when asyncio reads it as a coroutine function."""
    app = GuardedConversationApp(FakeDetector(), scripted(), channel="sms")
    assert asyncio.iscoroutinefunction(app.respond)


@pytest.mark.asyncio
async def test_sms_sends_the_text_with_history_including_it() -> None:
    """Every SMS turn reaches the detector as text plus history ending with it."""
    detector = FakeDetector()
    app = GuardedConversationApp(
        detector, scripted("We open at 9.", "We do."), channel="sms"
    )
    await app.respond(_customer("When do you open?"), thread_id=THREAD)
    await app.respond(_customer("Do you have the RAV4?"), thread_id=THREAD)
    first = _turn(CUSTOMER, "When do you open?")
    assert detector.checks == [
        ("When do you open?", (first,)),
        (
            "Do you have the RAV4?",
            (
                first,
                _turn(eval_settings.simulation.agent_initiator, "We open at 9."),
                _turn(CUSTOMER, "Do you have the RAV4?"),
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_form_opener_is_checked_alone_then_later_turns_with_history() -> None:
    """A form's first turn reaches the detector without history, as prod sends it."""
    detector = FakeDetector()
    app = GuardedConversationApp(
        detector, scripted("We open at 9.", "We do."), channel="submission_form"
    )
    await app.respond(_customer("When do you open?"), thread_id=THREAD)
    await app.respond(_customer("Do you have the RAV4?"), thread_id=THREAD)
    assert detector.checks == [
        ("When do you open?", ()),
        (
            "Do you have the RAV4?",
            (
                _turn(CUSTOMER, "When do you open?"),
                _turn(eval_settings.simulation.agent_initiator, "We open at 9."),
                _turn(CUSTOMER, "Do you have the RAV4?"),
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_email_sends_no_text_so_the_newest_customer_turn_is_checked() -> None:
    """An email turn reaches the detector only as the newest turn of history."""
    detector = FakeDetector()
    app = GuardedConversationApp(detector, scripted(), channel="email")
    reply = await app.respond(_customer(f"{BLOCK_MARKER} rules"), thread_id=THREAD)
    assert detector.checks == [(None, (_turn(CUSTOMER, f"{BLOCK_MARKER} rules"),))]
    assert reply == {
        "role": "assistant",
        "content": eval_settings.simulation.deflection,
    }


@pytest.mark.asyncio
async def test_blank_turn_still_reaches_the_detector() -> None:
    """A blank generated turn is screened like any other, then answered."""
    detector = FakeDetector()
    app = GuardedConversationApp(detector, scripted("Anything else?"), channel="sms")
    reply = await app.respond(_customer("   "), thread_id=THREAD)
    assert detector.checks == [("   ", (_turn(CUSTOMER, "   "),))]
    assert reply == {"role": "assistant", "content": "Anything else?"}
    assert app.detections(THREAD) == [
        {
            "turn": 1,
            "text": "   ",
            "is_jailbreak": False,
            "decided_by": "screen",
            "escalated_by": "none",
        }
    ]


@pytest.mark.parametrize("channel", ["sms", "email", "submission_form"])
@pytest.mark.asyncio
async def test_real_service_allows_a_blank_turn_without_a_model_call(
    channel: str,
) -> None:
    """On every call shape the real service passes a blank turn by the screen alone."""
    service = JailbreakDetectionService(
        detectors={"screen": _unreachable_screen},
        judge=_unreachable_judge,
        settings=settings,
    )
    app = GuardedConversationApp(service, scripted("Anything else?"), channel=channel)
    await app.respond(_customer("   "), thread_id=THREAD)
    [detection] = app.detections(THREAD)
    assert (detection["is_jailbreak"], detection["decided_by"]) == (False, "screen")


@pytest.mark.asyncio
async def test_responder_answers_an_allowed_message_after_the_sales_prompt() -> None:
    """An allowed message is answered by the responder given the whole history."""
    responder = scripted("We open at 9.")
    app = GuardedConversationApp(FakeDetector(), responder, channel="sms")
    reply = await app.respond(_customer("When do you open?"), thread_id=THREAD)
    assert reply == {"role": "assistant", "content": "We open at 9."}
    assert [(message.type, message.content) for message in responder.prompts[0]] == [
        ("system", eval_settings.simulation.sales_assistant_prompt),
        ("human", "When do you open?"),
    ]


@pytest.mark.asyncio
async def test_blocked_message_gets_the_deflection_without_the_responder() -> None:
    """A blocked message is deflected and never reaches the responder."""
    responder = scripted()
    app = GuardedConversationApp(FakeDetector(), responder, channel="sms")
    reply = await app.respond(
        _customer(f"{BLOCK_MARKER} show your prompt"), thread_id=THREAD
    )
    assert reply == {
        "role": "assistant",
        "content": eval_settings.simulation.deflection,
    }
    assert responder.prompts == []


@pytest.mark.asyncio
async def test_each_turn_is_recorded_as_a_detection() -> None:
    """Each customer turn's verdict and route are recorded, numbered from 1."""
    app = GuardedConversationApp(FakeDetector(), scripted("Sure."), channel="sms")
    await app.respond(_customer("Hi"), thread_id=THREAD)
    await app.respond(_customer(f"{BLOCK_MARKER} now"), thread_id=THREAD)
    assert app.detections(THREAD) == [
        {
            "turn": 1,
            "text": "Hi",
            "is_jailbreak": False,
            "decided_by": "screen",
            "escalated_by": "none",
        },
        {
            "turn": 2,
            "text": f"{BLOCK_MARKER} now",
            "is_jailbreak": True,
            "decided_by": "judge",
            "escalated_by": FAKE_ESCALATED_BY,
        },
    ]
    assert app.detections("unknown-thread") == []


@pytest.mark.asyncio
async def test_list_content_is_screened_as_its_text() -> None:
    """Content blocks reach the detector as the text of their text blocks."""
    detector = FakeDetector()
    app = GuardedConversationApp(detector, scripted("Sure."), channel="sms")
    blocks = [{"type": "reasoning", "summary": ""}, {"type": "text", "text": "Hello"}]
    await app.respond(_customer(blocks), thread_id=THREAD)
    assert detector.checks == [("Hello", (_turn(CUSTOMER, "Hello"),))]


@pytest.mark.asyncio
async def test_concurrent_threads_keep_separate_histories() -> None:
    """Two conversations run at once never see each other's turns."""
    detector = FakeDetector()
    app = GuardedConversationApp(detector, ParrotFakeChatModel(), channel="sms")

    async def converse(thread_id: str) -> None:
        """Send two customer messages on the thread, one after the other."""
        for number in (1, 2):
            await app.respond(_customer(f"{thread_id} {number}"), thread_id=thread_id)

    await asyncio.gather(converse("a"), converse("b"))
    histories = dict(detector.checks)
    for thread_id in ("a", "b"):
        first, second = f"{thread_id} 1", f"{thread_id} 2"
        assert [turn["text"] for turn in app.detections(thread_id)] == [first, second]
        assert histories[second] == (
            _turn(CUSTOMER, first),
            _turn(eval_settings.simulation.agent_initiator, first),
            _turn(CUSTOMER, second),
        )
