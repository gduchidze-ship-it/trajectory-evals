"""Multi-turn simulations and red-team attacks against an agent under test."""

from trajectory_evals.simulation.app import A2AConversationApp
from trajectory_evals.simulation.cases import Scenario, ScenarioSource, redteam_cases
from trajectory_evals.simulation.evaluators import (
    create_async_attack_succeeded_evaluator,
    create_async_turns_to_break_evaluator,
)
from trajectory_evals.simulation.judge import AttackJudge
from trajectory_evals.simulation.personas import AgentPersona, Persona, load_personas
from trajectory_evals.simulation.settings import (
    RedteamSection,
    SimulationSection,
    agent_table,
)
from trajectory_evals.simulation.target import ModelFactory, SimulationTarget, StopWhen
from trajectory_evals.simulation.types import (
    PersonaKind,
    ScenarioInputs,
    SimulationOutputs,
)
from trajectory_evals.simulation.user import (
    SimulatedUser,
    create_simulated_user,
    user_system_prompt,
)

__all__ = [
    "A2AConversationApp",
    "AgentPersona",
    "AttackJudge",
    "ModelFactory",
    "Persona",
    "PersonaKind",
    "RedteamSection",
    "Scenario",
    "ScenarioInputs",
    "ScenarioSource",
    "SimulatedUser",
    "SimulationOutputs",
    "SimulationSection",
    "SimulationTarget",
    "StopWhen",
    "agent_table",
    "create_async_attack_succeeded_evaluator",
    "create_async_turns_to_break_evaluator",
    "create_simulated_user",
    "load_personas",
    "redteam_cases",
    "user_system_prompt",
]
