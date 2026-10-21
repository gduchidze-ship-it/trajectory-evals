"""The aevaluate target that runs one guarded multi-turn simulation per example."""

import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from openevals.simulators import run_multiturn_simulation_async

from trajectory_evals.suites.jailbreak.protocols import Detector
from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.simulation.app import GuardedConversationApp
from trajectory_evals.suites.jailbreak.simulation.attempt import AttackAttemptJudge
from trajectory_evals.suites.jailbreak.simulation.personas import PERSONAS
from trajectory_evals.suites.jailbreak.simulation.user import create_simulated_user
from trajectory_evals.suites.jailbreak.types import (
    ChatCompletionMessage,
    SimulationInputs,
    SimulationOutputs,
)


class SimulationTarget:
    """Continues each example's opener in its persona until a block or max_turns."""

    def __init__(
        self,
        *,
        detector: Detector,
        user_client: BaseChatModel,
        responder: BaseChatModel,
        judge: BaseChatModel,
        max_turns: int = settings.simulation.max_turns,
    ) -> None:
        """Hold the collaborators every conversation shares; reject max_turns < 1."""
        if max_turns < 1:
            message = settings.texts.at_least_one
            raise ValueError(message.format(name="max_turns", value=max_turns))
        self._detector = detector
        self._user_client = user_client
        self._responder = responder
        self._judge = AttackAttemptJudge(judge)
        self._max_turns = max_turns

    async def __call__(self, inputs: SimulationInputs) -> SimulationOutputs:
        """Run the conversation with a fresh app, then judge its generated turns."""
        channel = inputs["channel"]
        app = GuardedConversationApp(self._detector, self._responder, channel=channel)
        thread_id = str(uuid.uuid4())

        async def blocked(_trajectory: list[ChatCompletionMessage], **_: Any) -> bool:
            """Return whether the detector has blocked a turn of this conversation."""
            return any(turn["is_jailbreak"] for turn in app.detections(thread_id))

        result = await run_multiturn_simulation_async(
            app=app.respond,
            user=create_simulated_user(
                persona=PERSONAS[inputs["persona"]],
                opening_message=inputs["opening_message"],
                channel=channel,
                max_turns=self._max_turns,
                client=self._user_client,
            ),
            max_turns=self._max_turns,
            stopping_condition=blocked,
            thread_id=thread_id,
        )
        detections = app.detections(thread_id)
        return SimulationOutputs(
            trajectory=result["trajectory"],
            detections=detections,
            attack_attempted=await self._judge(detections),
        )
