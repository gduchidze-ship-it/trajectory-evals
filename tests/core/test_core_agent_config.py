"""Tests for loading an agent folder's agent.toml and the files it names."""

import re
from pathlib import Path

import pytest

from trajectory_evals.core.agent_config import (
    AgentConfig,
    AgentConfigError,
    AgentSection,
    load_agent,
)

AGENT = '[agent]\nurl = "http://localhost:9999"\ndescription = "A car sales agent."\n'
SIMULATION = '[simulation]\nuser_prompt = "prompts/user.md"\nmax_turns = 4\n'
JUDGES = '[judges.helpful]\nprompt = "prompts/helpful.md"\n'
GRADERS = '[graders]\nfactory = "examples.sales.graders:build"\n'
LANGSMITH = '[langsmith]\nproject = "sales-prod"\n'
SECRET = "sk-live-4111"


def _agent_dir(agents_dir: Path, toml: str, name: str = "sales") -> Path:
    """Write agents_dir/name/agent.toml holding toml and return the folder."""
    root = agents_dir / name
    root.mkdir(parents=True)
    (root / "agent.toml").write_text(toml, encoding="utf-8")
    return root


def test_name_loads_the_folder_with_defaults_and_raw_tables(tmp_path: Path) -> None:
    """A bare name reads agents_dir/<name>/agent.toml; absent tables read as empty."""
    root = _agent_dir(tmp_path, AGENT + SIMULATION + JUDGES + GRADERS + LANGSMITH)

    config = load_agent("sales", agents_dir=tmp_path)

    assert config == AgentConfig(
        name="sales",
        root=root.resolve(),
        agent=AgentSection(
            url="http://localhost:9999",
            auth_token_env=None,
            timeout_s=120,
            history="metadata",
            description="A car sales agent.",
        ),
        bridge=None,
        sections={
            "simulation": {"user_prompt": "prompts/user.md", "max_turns": 4},
            "judges": {"helpful": {"prompt": "prompts/helpful.md"}},
            "graders": {"factory": "examples.sales.graders:build"},
            "langsmith": {"project": "sales-prod"},
        },
    )
    assert config.section("judges") == {"helpful": {"prompt": "prompts/helpful.md"}}
    assert config.section("langsmith") == {"project": "sales-prod"}
    assert config.section("redteam") == {}


def test_bridge_only_agent_loads_from_a_folder_path(tmp_path: Path) -> None:
    """A path is the folder itself, and [bridge] stands in for a url."""
    toml = (
        '[agent]\ndescription = "Wrapped."\n'
        '[bridge]\nresponder = "trajectory_evals.connect.example:respond"\n'
    )
    root = _agent_dir(tmp_path / "elsewhere", toml, name="wrapped")

    config = load_agent(root, agents_dir=tmp_path / "unused")

    assert (config.name, config.agent.url) == ("wrapped", None)
    assert config.bridge is not None
    assert config.bridge.responder == "trajectory_evals.connect.example:respond"
    assert config.bridge.options == {}


def test_a_str_with_a_separator_is_a_folder_path(tmp_path: Path) -> None:
    """Only a one-part str is looked up as a name under agents_dir."""
    root = _agent_dir(tmp_path, AGENT)
    assert load_agent(str(root), agents_dir=tmp_path / "unused").root == root.resolve()


BAD_FILES = {
    "no_agent_table": (SIMULATION, "agent: Field required"),
    "unknown_table": (AGENT + "[tools]\nx = 1\n", "tools: Extra inputs"),
    "no_url_or_bridge": (
        '[agent]\ndescription = "x"\n',
        "[agent] needs a url, or the file needs a [bridge] table",
    ),
    "zero_timeout": (AGENT + "timeout_s = 0\n", "agent.timeout_s: Input should be"),
    "negative_timeout": (AGENT + "timeout_s = -1\n", "agent.timeout_s: "),
    "unknown_agent_key": (AGENT + "retries = 2\n", "agent.retries: Extra inputs"),
    "no_description": ('[agent]\nurl = "http://a"\n', "agent.description: Field"),
    "bad_history": (AGENT + 'history = "full"\n', "agent.history: Input should be"),
    "bad_responder": (
        AGENT + '[bridge]\nresponder = "no_colon"\n',
        "bridge.responder: String should match",
    ),
    "raw_table_not_a_table": (
        "simulation = 3\n" + AGENT,
        "simulation: Input should be a valid dictionary",
    ),
    "invalid_toml": ("[agent\n", "Expected"),
}


@pytest.mark.parametrize(("toml", "reason"), BAD_FILES.values(), ids=BAD_FILES.keys())
def test_bad_agent_file_names_the_file_and_reason(
    tmp_path: Path, toml: str, reason: str
) -> None:
    """Every invalid agent.toml raises AgentConfigError naming the file."""
    root = _agent_dir(tmp_path, toml)
    pattern = f"^{re.escape(str(root.resolve() / 'agent.toml'))}: .*{re.escape(reason)}"
    with pytest.raises(AgentConfigError, match=pattern):
        load_agent("sales", agents_dir=tmp_path)


def test_missing_agent_names_the_file_it_looked_for(tmp_path: Path) -> None:
    """An unknown agent name fails with the agent.toml path it tried."""
    expected = re.escape(str(tmp_path.resolve() / "ghost" / "agent.toml"))
    with pytest.raises(AgentConfigError, match=f"file not found: {expected}"):
        load_agent("ghost", agents_dir=tmp_path)


def test_errors_never_echo_values(tmp_path: Path) -> None:
    """A bad value is described by its location, never repeated."""
    _agent_dir(tmp_path, AGENT + f'timeout_s = "{SECRET}"\n')
    with pytest.raises(AgentConfigError) as raised:
        load_agent("sales", agents_dir=tmp_path)
    assert "agent.timeout_s" in str(raised.value)
    assert SECRET not in str(raised.value)
    assert raised.value.__cause__ is None


def test_read_text_reads_a_file_under_the_folder(tmp_path: Path) -> None:
    """A relative path inside the folder returns that file's text."""
    root = _agent_dir(tmp_path, AGENT)
    (root / "prompts").mkdir()
    (root / "prompts" / "system.md").write_text("Be brief.", encoding="utf-8")
    assert load_agent("sales", agents_dir=tmp_path).read_text("prompts/system.md") == (
        "Be brief."
    )


def test_read_text_refuses_a_path_that_leaves_the_folder(tmp_path: Path) -> None:
    """Parent hops, absolute paths and outward symlinks are all refused."""
    root = _agent_dir(tmp_path, AGENT)
    outside = tmp_path / "outside.md"
    outside.write_text(SECRET, encoding="utf-8")
    (root / "link.md").symlink_to(outside)
    config = load_agent("sales", agents_dir=tmp_path)

    for relative in ("../outside.md", str(outside), "link.md"):
        with pytest.raises(AgentConfigError, match="is outside the agent folder"):
            config.read_text(relative)


def test_read_text_of_a_missing_file_names_it(tmp_path: Path) -> None:
    """A missing file inside the folder fails with its path."""
    _agent_dir(tmp_path, AGENT)
    config = load_agent("sales", agents_dir=tmp_path)
    with pytest.raises(AgentConfigError, match="file not found: .*prompts/none\\.md"):
        config.read_text("prompts/none.md")
