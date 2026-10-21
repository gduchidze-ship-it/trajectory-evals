"""The simulated user: a persona told its scenario, the rules and when to stop."""

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from openevals.simulators import create_async_llm_simulated_user

from trajectory_evals.core.types import ChatCompletionMessage
from trajectory_evals.simulation.personas import Persona
from trajectory_evals.simulation.settings import settings

SimulatedUser = Callable[..., Awaitable[ChatCompletionMessage]]


def user_system_prompt(
    persona: Persona,
    scenario: str,
    *,
    user_prompt: str | None,
    brief: str | None,
    stop_token: str,
) -> str:
    """Return the system prompt of persona in scenario, its stop token included.

    An attack persona's prompt opens with brief, or the library brief if None, and
    tells it to keep attacking and stop only once it succeeds; a benign persona may
    also stop when it gives up.
    """
    prompts = settings.prompts
    attack = persona.kind == "attack"
    attack_brief = (brief or prompts.redteam_brief,) if attack else ()
    stop = prompts.attack_stop_instruction if attack else prompts.stop_instruction
    parts = (
        *attack_brief,
        *((user_prompt,) if user_prompt else ()),
        persona.prompt,
        prompts.scenario.format(scenario=scenario),
        prompts.conversation_rules,
        stop.format(stop_token=stop_token),
    )
    return prompts.paragraph_separator.join(parts)


def create_simulated_user(
    system: str, *, opening_message: str | None, client: BaseChatModel
) -> SimulatedUser:
    """Create an openevals user whose first turn is opening_message verbatim, if any."""
    fixed_responses: list[str | ChatCompletionMessage] | None = (
        None if opening_message is None else [opening_message]
    )
    user: SimulatedUser = create_async_llm_simulated_user(
        system=system, client=client, fixed_responses=fixed_responses
    )
    return user


__all__ = ["SimulatedUser", "create_simulated_user", "user_system_prompt"]
