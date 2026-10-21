"""Simulated customers that continue a real conversation, benign or attacking."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.types import PersonaKind


@dataclass(frozen=True)
class Persona:
    """A simulated customer: its registry name, whether it attacks, and its prompt."""

    name: str
    kind: PersonaKind
    system_prompt: str


def _system_prompt(kind: PersonaKind, prompt: str) -> str:
    """Return a persona's prompt, after the red-team brief if it attacks, then rules."""
    texts = settings.simulation
    brief = (texts.red_team_brief,) if kind == "attack" else ()
    return texts.paragraph_separator.join((*brief, prompt, texts.conversation_rules))


PERSONAS: Mapping[str, Persona] = MappingProxyType(
    {
        name: Persona(name, persona.kind, _system_prompt(persona.kind, persona.prompt))
        for name, persona in settings.personas.items()
    }
)
