"""Multi-turn simulations in which a persona continues a real customer conversation."""

from trajectory_evals.suites.jailbreak.simulation.app import GuardedConversationApp
from trajectory_evals.suites.jailbreak.simulation.cases import SimulationCaseSource
from trajectory_evals.suites.jailbreak.simulation.evaluators import (
    create_async_attack_attempted_evaluator,
    create_async_persona_faithful_evaluator,
    create_async_simulation_correct_evaluator,
    create_async_turns_to_block_evaluator,
    create_attack_attempted_evaluator,
    create_persona_faithful_evaluator,
    create_simulation_correct_evaluator,
    create_turns_to_block_evaluator,
)
from trajectory_evals.suites.jailbreak.simulation.personas import PERSONAS, Persona
from trajectory_evals.suites.jailbreak.simulation.summary import create_simulation_summary_evaluators
from trajectory_evals.suites.jailbreak.simulation.target import SimulationTarget
from trajectory_evals.suites.jailbreak.simulation.user import create_simulated_user

__all__ = [
    "PERSONAS",
    "GuardedConversationApp",
    "Persona",
    "SimulationCaseSource",
    "SimulationTarget",
    "create_async_attack_attempted_evaluator",
    "create_async_persona_faithful_evaluator",
    "create_async_simulation_correct_evaluator",
    "create_async_turns_to_block_evaluator",
    "create_attack_attempted_evaluator",
    "create_persona_faithful_evaluator",
    "create_simulated_user",
    "create_simulation_correct_evaluator",
    "create_simulation_summary_evaluators",
    "create_turns_to_block_evaluator",
]
