"""Connect settings: config.toml values, overridable by environment."""

import os
from pathlib import Path
from typing import Self

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
AUTHORIZATION = "Authorization"


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
    bad_trajectory: Text
    timed_out: Text
    other_origin: Text
    no_bridge: Text
    responder_failed: Text
    responder_failed_log: Text
    no_reply: Text


class ConnectSettings(PackageSettings):
    """Connect settings; .env entries without the prefix are ignored."""

    model_config = SettingsConfigDict(
        toml_file=CONFIG_PATH, env_prefix="TRAJECTORY_EVALS_CONNECT_"
    )

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
                    {"turn", "index"},
                ),
                "texts.missing_env": (texts.missing_env, {"name"}),
                "texts.reserved_context": (texts.reserved_context, {"keys"}),
                "texts.history_not_applied": (texts.history_not_applied, set()),
                "texts.overrides_not_applied": (texts.overrides_not_applied, set()),
                "texts.task_not_completed": (
                    texts.task_not_completed,
                    {"state", "text"},
                ),
                "texts.bad_trajectory": (texts.bad_trajectory, set()),
                "texts.timed_out": (texts.timed_out, {"timeout_s"}),
                "texts.other_origin": (texts.other_origin, {"configured", "named"}),
                "texts.no_bridge": (texts.no_bridge, {"name"}),
                "texts.responder_failed": (texts.responder_failed, set()),
                "texts.no_reply": (texts.no_reply, {"path"}),
            }
        )
        return self


settings = ConnectSettings()


def _bearer_headers(token_env: str | None) -> dict[str, str]:
    """Return the Authorization header holding the named variable's token, if any.

    Raises AgentError naming the variable when it is unset or blank.
    """
    if token_env is None:
        return {}
    # A CRLF .env leaves "\r" on the token, and httpx echoes an illegal header value.
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise AgentError(settings.texts.missing_env.format(name=token_env))
    return {AUTHORIZATION: f"Bearer {token}"}


__all__ = [
    "AUTHORIZATION",
    "CONFIG_PATH",
    "PACKAGE_DIR",
    "BridgeSettings",
    "ConnectSettings",
    "SequenceRunnerSettings",
    "TextSettings",
    "_bearer_headers",
    "settings",
]
