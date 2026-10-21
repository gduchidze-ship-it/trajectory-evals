"""Simulation settings, the persona library, and the agent.toml tables they govern."""

from pathlib import Path
from typing import Self

from pydantic import Field, PositiveInt, TypeAdapter, ValidationError, model_validator
from pydantic_settings import SettingsConfigDict

from trajectory_evals.core.agent_config import (
    AGENT_FILE,
    AgentConfig,
    AgentConfigError,
    ImportPath,
)
from trajectory_evals.core.jsonl import _describe
from trajectory_evals.core.settings import (
    PackageSettings,
    RawText,
    Text,
    _require_placeholders,
    _Section,
)
from trajectory_evals.core.settings import settings as core_settings
from trajectory_evals.simulation.types import PersonaKind

PACKAGE_DIR = Path(__file__).parent
CONFIG_PATH = PACKAGE_DIR / "config.toml"
PERSONAS_PATH = PACKAGE_DIR / "personas.toml"


class DefaultSettings(_Section):
    """What a run uses when neither the command line, the case nor agent.toml says."""

    user_model: Text
    judge_model: Text
    max_turns: PositiveInt
    stop_token: Text
    attempts: PositiveInt


class KeySettings(_Section):
    """Feedback keys, the trace tag, and the red-team cases' source name and id."""

    ls_framework: Text
    attack_succeeded: Text
    defended: Text
    turns_to_break: Text
    redteam_source: Text
    redteam_case_id: Text


class PromptSettings(_Section):
    """The simulated user's system prompt parts, and the attack judge's prompt."""

    paragraph_separator: RawText
    redteam_brief: Text
    scenario: Text
    conversation_rules: Text
    stop_instruction: Text
    attack_stop_instruction: Text
    stopped_reply: Text
    redteam_scenario: Text
    attack_judge: Text


class TextSettings(_Section):
    """Comments on unscored rows, and messages for bad configuration."""

    target_errored: Text
    benign: Text
    not_broken: Text
    not_judged: Text
    judge_failed: Text
    section_error: Text
    one_prompt: Text
    unknown_persona: Text
    not_attack_personas: Text
    no_policy: Text


class PersonaSettings(_Section):
    """A library persona: whether it attacks, and its prompt about the agent."""

    kind: PersonaKind
    prompt: Text


class SimulationSettings(PackageSettings):
    """All simulation settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=(CONFIG_PATH, PERSONAS_PATH),
        env_prefix="TRAJECTORY_EVALS_SIMULATION_",
    )

    defaults: DefaultSettings
    keys: KeySettings
    prompts: PromptSettings
    texts: TextSettings
    personas: dict[Text, PersonaSettings] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_templates(self) -> Self:
        """Reject a template whose placeholders differ from those the code fills in."""
        keys, prompts, texts = self.keys, self.prompts, self.texts
        conversation = {"policy", "conversation", "reply"}
        stop = {"stop_token"}
        _require_placeholders(
            {
                "keys.redteam_source": (keys.redteam_source, {"digest"}),
                "keys.redteam_case_id": (keys.redteam_case_id, {"persona", "attempt"}),
                "prompts.scenario": (prompts.scenario, {"scenario"}),
                "prompts.stop_instruction": (prompts.stop_instruction, stop),
                "prompts.attack_stop_instruction": (
                    prompts.attack_stop_instruction,
                    stop,
                ),
                "prompts.attack_judge": (prompts.attack_judge, conversation),
                "texts.not_judged": (texts.not_judged, {"errors"}),
                "texts.section_error": (texts.section_error, {"section", "message"}),
                "texts.unknown_persona": (texts.unknown_persona, {"name", "known"}),
                "texts.not_attack_personas": (
                    texts.not_attack_personas,
                    {"names", "known"},
                ),
                "texts.no_policy": (texts.no_policy, {"persona"}),
                **{
                    f"personas.{name}.prompt": (persona.prompt, {"agent_description"})
                    for name, persona in self.personas.items()
                },
            }
        )
        return self


class SimulationSection(_Section):
    """agent.toml [simulation]: the simulated user and when a conversation ends.

    user_prompt is a file under the agent folder whose text opens every
    simulated user's system prompt. stop_when is a module:attr predicate over the
    transcript so far that ends the conversation when it returns True.
    """

    user_prompt: Text | None = None
    user_model: Text = Field(default_factory=lambda: settings.defaults.user_model)
    max_turns: PositiveInt = Field(default_factory=lambda: settings.defaults.max_turns)
    stop_token: Text = Field(default_factory=lambda: settings.defaults.stop_token)
    stop_when: ImportPath | None = None


class RedteamSection(_Section):
    """agent.toml [redteam]: the attackers' brief, the policy judged, and the cases.

    brief and policy are files under the agent folder. personas names the attack
    personas to run, every attack persona when absent, attempts times each.
    """

    brief: Text | None = None
    policy: Text | None = None
    personas: tuple[Text, ...] | None = Field(default=None, min_length=1)
    attempts: PositiveInt = Field(default_factory=lambda: settings.defaults.attempts)
    judge_model: Text = Field(default_factory=lambda: settings.defaults.judge_model)


def agent_table[Table](config: AgentConfig, name: str, schema: type[Table]) -> Table:
    """Validate config's raw table name as schema; AgentConfigError names the file.

    An absent table validates as {}, so schema's defaults apply. KeyError if name
    is not an agent.toml table.
    """
    try:
        return TypeAdapter(schema).validate_python(config.section(name))
    except ValidationError as error:
        described = _describe(error)
        message = settings.texts.section_error.format(section=name, message=described)
        path = config.root / AGENT_FILE
        raise AgentConfigError(
            core_settings.texts.file_error.format(path=path, message=message)
        ) from None


# Annotated because the section defaults above read it before mypy reaches here.
settings: SimulationSettings = SimulationSettings()

__all__ = [
    "CONFIG_PATH",
    "PACKAGE_DIR",
    "PERSONAS_PATH",
    "DefaultSettings",
    "KeySettings",
    "PersonaSettings",
    "PromptSettings",
    "RedteamSection",
    "SimulationSection",
    "SimulationSettings",
    "TextSettings",
    "agent_table",
    "settings",
]
