"""Connect settings: config.toml values, overridable by environment."""

import os
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

from trajectory_evals.core.settings import (
    PackageSettings,
    Text,
    _require_placeholders,
    _Section,
)
from trajectory_evals.core.types import AgentError

PACKAGE_DIR = Path(__file__).parent
CONFIG_PATH = PACKAGE_DIR / "config.toml"


class WireSettings(_Section):
    """The A2A metadata keys and artifact names agents outside this repo rely on."""

    history: Literal["history"]
    prompt_overrides: Literal["prompt_overrides"]
    applied: Literal["applied"]
    trajectory: Literal["trajectory"]
    messages: Literal["messages"]

    @property
    def reserved(self) -> frozenset[str]:
        """Return the request metadata keys a context key cannot use."""
        return frozenset({self.history, self.prompt_overrides})


class BridgeSettings(_Section):
    """The bridge's in-process url, card version and text media type."""

    url: Text
    version: Text
    media_type: Text


class SequenceRunnerSettings(_Section):
    """The runner's execute path and the id template of each mapped tool call."""

    execute_path: Text
    tool_call_id: Text


class TextSettings(_Section):
    """Messages for misuse and agent failures, and the bridge's log line."""

    missing_env: Text
    reserved_context: Text
    history_not_applied: Text
    overrides_not_applied: Text
    task_not_completed: Text
    no_bridge: Text
    responder_failed: Text
    responder_failed_log: Text
    one_url: Text
    no_reply: Text


class ConnectSettings(PackageSettings):
    """Connect settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH, env_prefix="TRAJECTORY_EVALS_CONNECT_"
    )

    wire: WireSettings
    bridge: BridgeSettings
    sequence_runner: SequenceRunnerSettings
    texts: TextSettings

    @model_validator(mode="after")
    def _check_templates(self) -> Self:
        """Reject a template whose placeholders differ from those the code fills in."""
        texts = self.texts
        _require_placeholders(
            {
                "sequence_runner.tool_call_id": (
                    self.sequence_runner.tool_call_id,
                    {"index"},
                ),
                "texts.missing_env": (texts.missing_env, {"name"}),
                "texts.reserved_context": (texts.reserved_context, {"keys"}),
                "texts.history_not_applied": (texts.history_not_applied, set()),
                "texts.overrides_not_applied": (texts.overrides_not_applied, set()),
                "texts.task_not_completed": (
                    texts.task_not_completed,
                    {"state", "text"},
                ),
                "texts.no_bridge": (texts.no_bridge, {"name"}),
                "texts.responder_failed": (texts.responder_failed, set()),
                "texts.one_url": (texts.one_url, set()),
                "texts.no_reply": (texts.no_reply, {"path"}),
            }
        )
        return self


settings = ConnectSettings()


def _require_env(name: str) -> str:
    """Return the environment variable's value; AgentError names it when unset."""
    value = os.environ.get(name)
    if value is None:
        raise AgentError(settings.texts.missing_env.format(name=name))
    return value


def _bearer_headers(token_env: str | None) -> dict[str, str]:
    """Return the Authorization header holding the named variable's token, if any."""
    if token_env is None:
        return {}
    return {"Authorization": f"Bearer {_require_env(token_env)}"}


__all__ = [
    "CONFIG_PATH",
    "PACKAGE_DIR",
    "BridgeSettings",
    "ConnectSettings",
    "SequenceRunnerSettings",
    "TextSettings",
    "WireSettings",
    "_bearer_headers",
    "_require_env",
    "settings",
]
