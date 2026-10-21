"""Simulated users: the persona library and an agent's own [personas.*] tables."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self, cast

from pydantic import model_validator

from trajectory_evals.core.agent_config import AgentConfig
from trajectory_evals.core.settings import Text, _Section
from trajectory_evals.simulation.settings import agent_table, settings
from trajectory_evals.simulation.types import PersonaKind


@dataclass(frozen=True)
class Persona:
    """A simulated user: its name, whether it attacks, and its prompt."""

    name: str
    kind: PersonaKind
    prompt: str


class AgentPersona(_Section):
    """agent.toml [personas.<name>]: its kind, and its prompt or a file holding it."""

    kind: PersonaKind
    prompt: Text | None = None
    prompt_file: Text | None = None

    @model_validator(mode="after")
    def _one_prompt(self) -> Self:
        """Require exactly one of prompt and prompt_file."""
        if (self.prompt is None) == (self.prompt_file is None):
            raise ValueError(settings.texts.one_prompt)
        return self


def load_personas(config: AgentConfig) -> Mapping[str, Persona]:
    """Return the library about config's agent, then its own personas, which win.

    Library prompts get the agent's description; the agent's prompts are its own
    text, used as written. Raises AgentConfigError for a bad [personas.<name>]
    table or a prompt_file it cannot read.
    """
    description = config.agent.description
    library = {
        name: Persona(
            name, persona.kind, persona.prompt.format(agent_description=description)
        )
        for name, persona in settings.personas.items()
    }
    tables = agent_table(config, "personas", dict[Text, AgentPersona])
    own = {
        name: Persona(name, table.kind, _prompt(table, config))
        for name, table in tables.items()
    }
    return MappingProxyType(library | own)


def _prompt(table: AgentPersona, config: AgentConfig) -> str:
    """Return the table's inline prompt, or the text of its prompt_file."""
    if table.prompt_file is None:
        return cast(str, table.prompt)  # _one_prompt: prompt is set
    return config.read_text(table.prompt_file)


__all__ = ["AgentPersona", "Persona", "load_personas"]
