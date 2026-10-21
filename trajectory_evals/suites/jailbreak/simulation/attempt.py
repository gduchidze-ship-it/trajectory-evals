"""The LLM judgement of whether a simulated customer's generated turns attacked."""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from openevals.llm import create_async_llm_as_judge
from openevals.prompts import PROMPT_INJECTION_PROMPT
from openevals.utils import _chat_completion_messages_to_string

from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.types import ChatCompletionMessage, TurnDetection
from trajectory_evals.suites.jailbreak.utils import _has_text

logger = logging.getLogger(__name__)


class AttackAttemptJudge:
    """Asks an LLM judge whether the turns after the real opener attempted an attack."""

    def __init__(self, judge: BaseChatModel) -> None:
        """Build the prompt-injection judge over the given chat model."""
        self._judge: Callable[..., Awaitable[Any]] = create_async_llm_as_judge(
            prompt=PROMPT_INJECTION_PROMPT,
            judge=judge,
            feedback_key=settings.simulation.attack_attempted_key,
        )

    async def __call__(self, detections: Sequence[TurnDetection]) -> bool | None:
        """Return the generated turns' verdict, or None if none or the judge fails."""
        simulation = settings.simulation
        turns: list[ChatCompletionMessage] = [
            {"role": simulation.customer_role, "content": detection["text"]}
            for detection in detections[simulation.opener_turn :]
            if _has_text(detection["text"])
        ]
        if not turns:
            return None
        try:
            result = await self._judge(
                inputs=_chat_completion_messages_to_string(turns)
            )
        except Exception as e:
            logger.warning(settings.texts.judge_failed, type(e).__name__)
            return None
        return bool(result["score"])
