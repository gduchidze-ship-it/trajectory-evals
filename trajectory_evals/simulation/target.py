"""The aevaluate target that runs one simulated conversation with the agent per case."""

import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pkgutil import resolve_name
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from openevals.simulators import run_multiturn_simulation_async

from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.protocols import Agent
from trajectory_evals.core.settings import settings as core_settings
from trajectory_evals.core.types import ChatCompletionMessage
from trajectory_evals.simulation.app import A2AConversationApp
from trajectory_evals.simulation.judge import AttackJudge
from trajectory_evals.simulation.personas import Persona, load_personas
from trajectory_evals.simulation.settings import (
    RedteamSection,
    SimulationSection,
    agent_table,
    settings,
)
from trajectory_evals.simulation.types import ScenarioInputs, SimulationOutputs
from trajectory_evals.simulation.user import (
    SimulatedUser,
    create_simulated_user,
    user_system_prompt,
)

logger = logging.getLogger(__name__)

ModelFactory = Callable[[str], BaseChatModel]
# [simulation] stop_when: called with a copy of the transcript after each agent
# turn; True ends the conversation, and anything it raises fails the row.
StopWhen = Callable[[list[ChatCompletionMessage]], bool]


@dataclass
class _Verdicts:
    """An attack conversation's judging so far: the first broken turn, the failures."""

    broken_turn: int | None = None
    judge_errors: int = 0

    async def record(
        self, judge: AttackJudge, transcript: list[ChatCompletionMessage]
    ) -> None:
        """Judge the latest turn; a failed judgement is counted, logged, not broken."""
        try:
            broken = await judge(transcript)
        # Fail open: a judge outage must not error the row; judge_errors shows it.
        except Exception as error:
            logger.warning(settings.texts.judge_failed, type(error).__name__)
            self.judge_errors += 1
            return
        if broken:
            self.broken_turn = _turns(transcript)


class SimulationTarget:
    """Talks to the agent as each case's persona until a stop condition or max_turns.

    A conversation ends when the simulated user sends the stop token, when the
    agent's [simulation] stop_when predicate holds for the transcript, when the
    attack judge finds an attack persona broke the [redteam] policy, or after
    max_turns user messages: the max_turns argument, else the case's, else the
    agent's.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        config: AgentConfig,
        model_factory: ModelFactory = init_chat_model,
        prompt_overrides: Mapping[str, str] | None = None,
        max_turns: int | None = None,
    ) -> None:
        """Read the agent's tables and files and build its models; fail if invalid.

        model_factory builds [simulation] user_model, and [redteam] judge_model
        only when a policy is set. Raises ValueError for max_turns below 1,
        AgentConfigError for a bad table or a file it cannot read, and ImportError
        or AttributeError for a stop_when that does not resolve.
        """
        if max_turns is not None and max_turns < 1:
            message = core_settings.texts.at_least_one
            raise ValueError(message.format(name="max_turns", value=max_turns))
        simulation = agent_table(config, "simulation", SimulationSection)
        redteam = agent_table(config, "redteam", RedteamSection)
        self._agent = agent
        self._prompt_overrides = prompt_overrides or None
        self._max_turns = max_turns
        self._simulation = simulation
        self._personas = load_personas(config)
        self._stop_when: StopWhen | None = (
            None if simulation.stop_when is None else resolve_name(simulation.stop_when)
        )
        self._user_prompt = _read(config, simulation.user_prompt)
        self._brief = _read(config, redteam.brief)
        self._user_model = model_factory(simulation.user_model)
        self._judge = (
            None
            if redteam.policy is None
            else AttackJudge(
                model_factory(redteam.judge_model), config.read_text(redteam.policy)
            )
        )

    async def __call__(self, inputs: ScenarioInputs) -> SimulationOutputs:
        """Run the case's conversation, then report its transcript and how it ended.

        Raises ValueError, before any message, for an unknown persona or an attack
        persona when [redteam] has no policy. An AgentError, a simulated user
        failure or a stop_when error propagates; a judge failure is counted in
        judge_errors instead.
        """
        persona = self._persona(inputs["persona"])
        judge = self._attack_judge(persona)
        app = A2AConversationApp(
            self._agent,
            prompt_overrides=self._prompt_overrides,
            context=inputs["context"],
            stop_token=self._simulation.stop_token,
        )
        thread_id = str(uuid.uuid4())
        verdicts = _Verdicts()

        async def stop(_trajectory: list[ChatCompletionMessage], **_: Any) -> bool:
            """Return whether the conversation is over after the latest turn."""
            if app.ended(thread_id):
                return True
            transcript = app.transcript(thread_id)
            if judge is not None:
                await verdicts.record(judge, transcript)
            return verdicts.broken_turn is not None or self._hook_stops(transcript)

        max_turns = self._max_turns or inputs["max_turns"] or self._simulation.max_turns
        await run_multiturn_simulation_async(
            app=app.respond,
            user=self._user(persona, inputs),
            max_turns=max_turns,
            stopping_condition=stop,
            thread_id=thread_id,
        )
        outputs = app.outputs(thread_id)
        return SimulationOutputs(
            **outputs,
            persona=persona.name,
            kind=persona.kind,
            broken_turn=verdicts.broken_turn,
            turns=_turns(outputs["messages"]),
            judge_errors=verdicts.judge_errors,
        )

    def _persona(self, name: str) -> Persona:
        """Return the named persona; raise ValueError naming the known ones."""
        if name not in self._personas:
            message = settings.texts.unknown_persona
            raise ValueError(message.format(name=name, known=list(self._personas)))
        return self._personas[name]

    def _attack_judge(self, persona: Persona) -> AttackJudge | None:
        """Return the judge for an attack persona, None for a benign one."""
        if persona.kind == "benign":
            return None
        if self._judge is None:
            raise ValueError(settings.texts.no_policy.format(persona=persona.name))
        return self._judge

    def _user(self, persona: Persona, inputs: ScenarioInputs) -> SimulatedUser:
        """Return the simulated user playing persona in the case's scenario."""
        system = user_system_prompt(
            persona,
            inputs["scenario"],
            user_prompt=self._user_prompt,
            brief=self._brief,
            stop_token=self._simulation.stop_token,
        )
        return create_simulated_user(
            system, opening_message=inputs["opening_message"], client=self._user_model
        )

    def _hook_stops(self, transcript: list[ChatCompletionMessage]) -> bool:
        """Return whether the agent's stop_when predicate, if any, ends the run."""
        return self._stop_when is not None and self._stop_when(transcript)


def _read(config: AgentConfig, relative: str | None) -> str | None:
    """Return the text of the agent's file at relative, or None if it names none."""
    return None if relative is None else config.read_text(relative)


def _turns(transcript: Sequence[ChatCompletionMessage]) -> int:
    """Return how many user messages the agent answered in the transcript."""
    return sum(message["role"] == "user" for message in transcript)


__all__ = ["ModelFactory", "SimulationTarget", "StopWhen"]
