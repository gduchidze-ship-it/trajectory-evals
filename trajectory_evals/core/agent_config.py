"""An agent folder: agents/<name>/agent.toml and the files beside it."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, PositiveFloat, StringConstraints

from trajectory_evals.core.jsonl import _describe
from trajectory_evals.core.settings import REPO_ROOT, Text, _Section, settings

AGENTS_DIR = REPO_ROOT / "agents"
AGENT_FILE = "agent.toml"

ImportPath = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[\w.]+:[\w.]+$")
]


class AgentConfigError(ValueError):
    """An agent folder whose agent.toml, or a file it names, cannot be read."""


class AgentSection(_Section):
    """[agent]: where the agent is served, how it is called, and what it does."""

    url: Text | None = None
    auth_token_env: Text | None = None
    timeout_s: PositiveFloat = 120
    history: Literal["metadata", "replay"] = "metadata"
    description: Text


class BridgeSection(_Section):
    """[bridge]: the module:attr responder factory that wraps a non-A2A agent."""

    responder: ImportPath
    options: dict[str, Any] = Field(default_factory=dict)


class _AgentFile(_Section):
    """agent.toml's tables; each raw one is validated by the package that reads it."""

    agent: AgentSection
    bridge: BridgeSection | None = None
    simulation: dict[str, Any] | None = None
    judges: dict[str, Any] | None = None
    personas: dict[str, Any] | None = None
    redteam: dict[str, Any] | None = None
    optimize: dict[str, Any] | None = None
    variants: dict[str, Any] | None = None
    graders: dict[str, Any] | None = None
    langsmith: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentConfig:
    """One loaded agent folder: its name, root, endpoint tables and raw tables."""

    name: str
    root: Path
    agent: AgentSection
    bridge: BridgeSection | None
    sections: Mapping[str, Mapping[str, Any]]

    def section(self, name: str) -> Mapping[str, Any]:
        """Return the named raw table, or {} if absent; KeyError for an unknown name."""
        if name not in _AgentFile.model_fields:
            raise KeyError(name)
        return self.sections.get(name, {})

    def read_text(self, relative: str) -> str:
        """Return the UTF-8 text of a file under root, refusing one outside it."""
        root = self.root.resolve()
        path = (root / relative).resolve()
        texts = settings.texts
        if not path.is_relative_to(root):
            raise AgentConfigError(texts.outside_root.format(path=relative, root=root))
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise AgentConfigError(texts.not_found.format(path=path)) from None
        except (OSError, UnicodeDecodeError) as error:
            message = texts.file_error.format(path=path, message=_describe(error))
            raise AgentConfigError(message) from None


def load_agent(
    name_or_dir: str | Path, *, agents_dir: Path = AGENTS_DIR
) -> AgentConfig:
    """Load a bare name from agents_dir/<name>, or any other path as the folder."""
    root = _agent_root(name_or_dir, agents_dir)
    path = root / AGENT_FILE
    texts = settings.texts
    try:
        with path.open("rb") as file:
            agent_file = _AgentFile.model_validate(tomllib.load(file))
    except FileNotFoundError:
        raise AgentConfigError(texts.not_found.format(path=path)) from None
    except (OSError, ValueError) as error:
        message = texts.file_error.format(path=path, message=_describe(error))
        raise AgentConfigError(message) from None
    if agent_file.agent.url is None and agent_file.bridge is None:
        raise AgentConfigError(texts.no_endpoint.format(path=path))
    return AgentConfig(
        name=root.name,
        root=root,
        agent=agent_file.agent,
        bridge=agent_file.bridge,
        sections=agent_file.model_dump(exclude={"agent", "bridge"}, exclude_none=True),
    )


def _agent_root(name_or_dir: str | Path, agents_dir: Path) -> Path:
    """Return the resolved agent folder; only a str that is a bare name is a name."""
    path = Path(name_or_dir)
    if isinstance(name_or_dir, str) and name_or_dir == path.name:
        path = agents_dir / path
    return path.resolve()


__all__ = [
    "AGENTS_DIR",
    "AGENT_FILE",
    "AgentConfig",
    "AgentConfigError",
    "AgentSection",
    "BridgeSection",
    "load_agent",
]
