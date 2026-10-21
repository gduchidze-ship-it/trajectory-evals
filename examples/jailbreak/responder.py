"""A bridge Responder that screens each customer message with detect_jailbreak."""

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self, cast

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field, model_validator
from src.toolsets.jailbreak_detection._call_log import track_call
from src.toolsets.jailbreak_detection.settings import settings as detector_settings
from src.toolsets.jailbreak_detection.types import (
    ConversationTurn,
    DecidedBy,
    JailbreakDetectionResult,
)

from examples.jailbreak.settings import settings
from examples.jailbreak.types import DetectionResult, ExpectedAction
from trajectory_evals.connect.bridge import AgentReply, AgentRequest, Responder
from trajectory_evals.core.settings import REPO_ROOT, Text, _Section
from trajectory_evals.core.types import AgentError, ChatCompletionMessage, ToolMessage

type ModelFactory = Callable[[str], BaseChatModel]


class Detector(Protocol):
    """Screens one inbound message the way detect_jailbreak does."""

    async def check(
        self, text: str | None, history: Sequence[ConversationTurn] | None = None
    ) -> JailbreakDetectionResult:
        """Return the verdict on text, or on the newest customer turn of history."""


class JailbreakOptions(_Section):
    """[bridge].options: answer with the verdict, or as the sales assistant.

    reply false answers each message with its verdict, BLOCKED or ALLOWED. reply
    true answers a blocked message with the deflection, and any other with
    model's reply under the system prompt. system_prompt and deflection are
    files relative to the repo root; model is a provider:model string.
    """

    reply: bool = False
    model: Text = Field(default_factory=lambda: settings.responder.model)
    system_prompt: Text | None = None
    deflection: Text | None = None

    @model_validator(mode="after")
    def _check_reply(self) -> Self:
        """Require both prompt files when the responder replies."""
        if self.reply and (self.system_prompt is None or self.deflection is None):
            raise ValueError(settings.texts.reply_needs_prompts)
        return self


@dataclass(frozen=True)
class _Assistant:
    """The sales assistant that answers what the detector allowed."""

    model: BaseChatModel
    system_prompt: str
    deflection: str


@dataclass(frozen=True)
class _JailbreakResponder:
    """Screens each message, reports the tool call and result, then answers."""

    detector: Detector
    assistant: _Assistant | None

    async def __call__(self, request: AgentRequest) -> AgentReply:
        """Answer one message with its detect_jailbreak call, result and reply.

        Prompt overrides other than the assistant's system prompt are refused
        before the detector runs: the reply is empty and never acknowledges
        them, so the client raises.
        """
        if not self._applies(request.prompt_overrides):
            return AgentReply(text="")
        result = await _screen(self.detector, request)
        text = await self._answer(request, result["is_jailbreak"])
        call_id = settings.responder.tool_call_id.format(turn=_turn(request.history))
        final: ChatCompletionMessage = {"role": "assistant", "content": text}
        messages = (_tool_call(call_id, request), _tool_result(call_id, result), final)
        return AgentReply(
            text=text,
            messages=messages,
            prompt_overrides_applied=bool(request.prompt_overrides),
        )

    def _applies(self, overrides: Mapping[str, str]) -> bool:
        """Return whether the responder can use every one of the overrides."""
        if not overrides:
            return True
        system = settings.responder.system_override
        return self.assistant is not None and overrides.keys() == {system}

    async def _answer(self, request: AgentRequest, blocked: bool) -> str:
        """Return the verdict, the deflection, or the assistant's reply."""
        if self.assistant is None:
            verdict: ExpectedAction = "BLOCKED" if blocked else "ALLOWED"
            return verdict
        if blocked:
            return self.assistant.deflection
        system = request.prompt_overrides.get(
            settings.responder.system_override, self.assistant.system_prompt
        )
        conversation = [
            {"role": "system", "content": system},
            *(
                {"role": message["role"], "content": _text(message)}
                for message in _chat(request.history)
            ),
            {"role": "user", "content": request.message},
        ]
        response = await self.assistant.model.ainvoke(conversation)
        return response.text


def factory(
    options: Mapping[str, Any],
    *,
    detector: Detector | None = None,
    model_factory: ModelFactory = init_chat_model,
) -> Responder:
    """Return a Responder that screens every message before it answers.

    detector defaults to the real cascade on Bedrock, built now; model_factory
    builds the assistant's model when options reply. Invalid options raise
    pydantic's ValidationError, a prompt file that cannot be read raises
    OSError, and AWS credentials that do not resolve raise AgentError, since the
    detector would fail open and score every row allowed.
    """
    parsed = JailbreakOptions.model_validate(options)
    assistant = None
    if parsed.reply:
        assistant = _Assistant(
            model=model_factory(parsed.model),
            # _check_reply: both files are set when the responder replies.
            system_prompt=_read(cast(str, parsed.system_prompt)),
            deflection=_read(cast(str, parsed.deflection)),
        )
    return _JailbreakResponder(detector or _build_detector(), assistant)


async def _screen(detector: Detector, request: AgentRequest) -> DetectionResult:
    """Check the message as prod does, in a tracked call, and return the result."""
    channel = request.context.get(settings.channels.key)
    prior = [_conversation_turn(message) for message in _chat(request.history)]
    arguments = _detector_arguments(channel, request.message, prior)
    with track_call(None, request.context_id) as call:
        result = await detector.check(*arguments)
    return DetectionResult(
        is_jailbreak=result.is_jailbreak,
        decided_by=result.decided_by,
        escalated_by=call.escalated_by,
        failed_open=result.decided_by == DecidedBy.FAIL_POLICY.value,
    )


def _detector_arguments(
    channel: str | None, text: str | None, prior: Sequence[ConversationTurn]
) -> tuple[str | None, tuple[ConversationTurn, ...]]:
    """Return the text and history prod sends the detector for a customer turn."""
    channels = settings.channels
    if not prior and channel == channels.opener_text_only:
        return text, ()
    current = ConversationTurn(
        initiator=detector_settings.input.customer_initiator, content=text
    )
    history = (*prior, current)
    if channel == channels.conversation_only:
        return None, history
    return text, history


def _chat(history: Sequence[ChatCompletionMessage]) -> list[ChatCompletionMessage]:
    """Return the customer and agent text messages, without tool calls or results."""
    return [
        message
        for message in history
        if message["role"] == "user"
        or (message["role"] == "assistant" and not message.get("tool_calls"))
    ]


def _conversation_turn(message: ChatCompletionMessage) -> ConversationTurn:
    """Return a message as the detector's turn; an agent's name is its initiator."""
    if message["role"] == "user":
        initiator = detector_settings.input.customer_initiator
    else:
        named = cast(Mapping[str, Any], message).get("name")
        initiator = named or settings.responder.agent_initiator
    return ConversationTurn(initiator=initiator, content=_text(message))


def _text(message: ChatCompletionMessage) -> str:
    """Return the message's text, joining its text blocks when content is a list."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


def _turn(history: Sequence[ChatCompletionMessage]) -> int:
    """Return this message's number among the conversation's user messages."""
    return 1 + sum(message["role"] == "user" for message in history)


def _tool_call(call_id: str, request: AgentRequest) -> ChatCompletionMessage:
    """Return the assistant message calling the tool on the customer's message."""
    channel = request.context.get(settings.channels.key)
    arguments = json.dumps({"text": request.message, "channel": channel})
    function = {"name": settings.keys.tool, "arguments": arguments}
    call = {"id": call_id, "type": "function", "function": function}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def _tool_result(call_id: str, result: DetectionResult) -> ToolMessage:
    """Return the tool message holding the detector's result as JSON."""
    return {"role": "tool", "content": json.dumps(result), "tool_call_id": call_id}


def _read(relative: str) -> str:
    """Return the stripped text of a file relative to the repo root."""
    return (REPO_ROOT / relative).read_text(encoding="utf-8").strip()


def _has_aws_credentials() -> bool:
    """Return whether boto3 resolves AWS credentials, refreshing an SSO login."""
    try:
        credentials = boto3.Session().get_credentials()
        if credentials is not None:
            credentials.get_frozen_credentials()
    except (BotoCoreError, ClientError):
        return False
    return credentials is not None


def _build_detector() -> Detector:
    """Return the real detection cascade on a Bedrock client in the AWS region."""
    if not _has_aws_credentials():
        raise AgentError(settings.texts.no_aws_credentials)
    # Imported on use, so this module imports without aioboto3, which the boto3
    # pin excludes.
    from src.toolsets.jailbreak_detection.bedrock import BedrockClient
    from src.toolsets.jailbreak_detection.factory import (
        build_jailbreak_detection_service,
    )

    responder = settings.responder
    region = os.environ.get(responder.aws_region_env, responder.default_aws_region)
    # src is Any to mypy without the detector repo on MYPYPATH.
    return build_jailbreak_detection_service(BedrockClient(detector_settings, region))  # type: ignore[no-any-return]


__all__ = ["Detector", "JailbreakOptions", "ModelFactory", "factory"]
