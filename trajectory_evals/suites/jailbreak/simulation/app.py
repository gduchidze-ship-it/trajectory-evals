"""The openevals app: the detector screens each customer turn before anyone replies."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from src.toolsets.jailbreak_detection._call_log import track_call
from src.toolsets.jailbreak_detection.settings import settings as detector_settings
from src.toolsets.jailbreak_detection.types import ConversationTurn

from trajectory_evals.suites.jailbreak.protocols import Detector
from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.types import ChatCompletionMessage, TurnDetection
from trajectory_evals.suites.jailbreak.utils import _detector_arguments


class GuardedConversationApp:
    """Screens each customer turn with the detector, then deflects it or replies."""

    def __init__(
        self,
        detector: Detector,
        responder: BaseChatModel,
        *,
        channel: str,
        deflection: str = settings.simulation.deflection,
    ) -> None:
        """Hold the collaborators and start with no conversation on any thread."""
        self._detector = detector
        self._responder = responder
        self._channel = channel
        self._deflection = deflection
        self._histories: dict[str, list[ConversationTurn]] = {}
        self._detections: dict[str, list[TurnDetection]] = {}

    async def respond(
        self, message: ChatCompletionMessage, *, thread_id: str, **_: Any
    ) -> ChatCompletionMessage:
        """Answer one customer message on the thread, with the deflection if blocked."""
        history = self._histories.setdefault(thread_id, [])
        text = _text_of(message)
        detection = await self._screen(thread_id, text, tuple(history))
        self._detections.setdefault(thread_id, []).append(detection)
        history.append(
            ConversationTurn(
                initiator=detector_settings.input.customer_initiator, content=text
            )
        )
        if detection["is_jailbreak"]:
            reply = self._deflection
        else:
            reply = await self._reply(history)
        agent = settings.simulation.agent_initiator
        history.append(ConversationTurn(initiator=agent, content=reply))
        return {"role": settings.simulation.agent_role, "content": reply}

    def detections(self, thread_id: str) -> list[TurnDetection]:
        """Return the detection on each customer turn of the thread so far."""
        return list(self._detections.get(thread_id, ()))

    async def _screen(
        self, thread_id: str, text: str, prior: tuple[ConversationTurn, ...]
    ) -> TurnDetection:
        """Check the customer's text, blank or not, as prod does, in a tracked call."""
        arguments = _detector_arguments(self._channel, text, prior)
        with track_call(None, thread_id) as call:
            result = await self._detector.check(*arguments)
        return TurnDetection(
            turn=len(self.detections(thread_id)) + 1,
            text=text,
            is_jailbreak=result.is_jailbreak,
            decided_by=result.decided_by,
            escalated_by=call.escalated_by,
        )

    async def _reply(self, history: Sequence[ConversationTurn]) -> str:
        """Return the responder's sales-assistant answer to the conversation so far."""
        texts = settings.simulation
        system = {"role": texts.system_role, "content": texts.sales_assistant_prompt}
        response = await self._responder.ainvoke(
            [system, *(_as_chat_message(turn) for turn in history)]
        )
        return response.text


def _text_of(message: ChatCompletionMessage) -> str:
    """Return the message's text, joining its text blocks when content is a list."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


def _as_chat_message(turn: ConversationTurn) -> dict[str, str]:
    """Return the turn as a chat message from the responder's point of view."""
    roles = settings.simulation
    customer = turn.initiator == detector_settings.input.customer_initiator
    role = roles.customer_role if customer else roles.agent_role
    return {"role": role, "content": turn.content or ""}
