"""The simulated customer that opens with a real message, then plays a persona."""

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from openevals.simulators import create_async_llm_simulated_user

from evals.jailbreak.settings import settings
from evals.jailbreak.simulation.personas import Persona
from evals.jailbreak.types import ChatCompletionMessage

SimulatedUser = Callable[..., Awaitable[ChatCompletionMessage]]


def create_simulated_user(
    *,
    persona: Persona,
    opening_message: str,
    channel: str,
    max_turns: int,
    client: BaseChatModel,
) -> SimulatedUser:
    """Create an openevals user whose first turn is opening_message verbatim."""
    texts = settings.simulation
    sms = channel == settings.channels.sms
    register = texts.sms_register if sms else texts.email_register
    budget = texts.turn_budget.format(follow_ups=max_turns - texts.opener_turn)
    user: SimulatedUser = create_async_llm_simulated_user(
        system=texts.paragraph_separator.join(
            (persona.system_prompt, register, budget)
        ),
        client=client,
        fixed_responses=[opening_message],
    )
    return user
