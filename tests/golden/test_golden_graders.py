"""Tests for building an agent's graders from its [judges] and [graders] tables."""

import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from tests.golden.golden_fakes import ModelFactory, agent_config
from trajectory_evals.core.agent_config import AgentConfig, AgentConfigError
from trajectory_evals.core.evaluators import _async_evaluator
from trajectory_evals.golden.graders import Graders, build_graders, load_custom_graders

DETERMINISTIC = [
    "output_exact",
    "contains",
    "not_contains",
    "tools_called",
    "trajectory_match",
]
FACTORY = "tests.golden.test_golden_graders:build_custom"


async def _resolved(**kwargs: Any) -> bool:
    """Pass every run; the custom grader these tests add."""
    return True


def _summary(outputs: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Report how many runs there were; the custom summary these tests add."""
    return {"key": "runs", "score": len(outputs)}


def build_custom(config: AgentConfig) -> Graders:
    """The [graders] factory these tests name: one row and one summary evaluator."""
    resolved = _async_evaluator(_resolved, "resolved", ls_framework="test")
    return Graders(
        evaluators=(resolved,), summary_evaluators=(_summary,), pass_keys=("resolved",)
    )


def _not_graders(config: AgentConfig) -> object:
    """A [graders] factory that returns the wrong thing."""
    return [config.name]


def _names(graders: Graders) -> list[str]:
    """Return each evaluator's feedback key, in order."""
    return [getattr(evaluator, "__name__") for evaluator in graders.evaluators]


def test_defaults_are_the_deterministic_graders_and_criteria(tmp_path: Path) -> None:
    """With no tables, every built-in grader runs and gates pass_rate."""
    factory = ModelFactory()
    graders = build_graders(
        agent_config(tmp_path), model_factory=factory, criteria_model="openai:judge"
    )

    assert _names(graders) == [*DETERMINISTIC, "criteria"]
    assert graders.pass_keys == (*DETERMINISTIC, "criteria")
    assert graders.summary_evaluators == ()
    assert factory.names == ["openai:judge"]


def test_without_criteria_no_model_is_made_and_criteria_is_no_pass_key(
    tmp_path: Path,
) -> None:
    """A dataset with only deterministic expectations needs no LLM API key."""

    def no_model(name: str) -> Any:
        """Fail every call, so making any model fails the test."""
        raise AssertionError(f"made a model: {name}")

    graders = build_graders(
        agent_config(tmp_path), model_factory=no_model, criteria=False
    )

    assert _names(graders) == DETERMINISTIC
    assert graders.pass_keys == tuple(DETERMINISTIC)


def test_every_built_evaluator_names_inputs_so_aevaluate_passes_them(
    tmp_path: Path,
) -> None:
    """aevaluate passes only named arguments; without inputs the span would widen."""
    judges = {"done": {"prompt": "openevals:TASK_COMPLETION_PROMPT"}}
    config = agent_config(tmp_path, judges=judges)
    graders = build_graders(config, model_factory=ModelFactory())

    assert len(graders.evaluators) == len(DETERMINISTIC) + 2
    for evaluator in graders.evaluators:
        assert "inputs" in inspect.signature(evaluator).parameters


def test_a_judge_model_that_cannot_be_made_raises(tmp_path: Path) -> None:
    """A missing key or provider package fails the build, not every row."""

    def no_key(name: str) -> Any:
        """Fail as init_chat_model does without the provider's API key."""
        raise ValueError(f"no API key for {name}")

    with pytest.raises(ValueError, match="no API key for openai:gpt-5.5"):
        build_graders(agent_config(tmp_path), model_factory=no_key)


def test_judges_and_custom_graders_follow_and_only_gates_join_pass_keys(
    tmp_path: Path,
) -> None:
    """Agent judges, then the factory's graders; a gate judge is a pass key."""
    judges = {
        "done": {"prompt": "openevals:TASK_COMPLETION_PROMPT", "gate": True},
        "tone": {"prompt": "openevals:AGENT_TONE_PROMPT", "continuous": True},
    }
    config = agent_config(tmp_path, judges=judges, graders={"factory": FACTORY})

    graders = build_graders(config, model_factory=ModelFactory())

    assert _names(graders) == [*DETERMINISTIC, "criteria", "done", "tone", "resolved"]
    assert graders.pass_keys == (*DETERMINISTIC, "criteria", "done", "resolved")
    assert [summary.__name__ for summary in graders.summary_evaluators] == ["_summary"]


def test_without_a_graders_table_there_are_no_custom_graders(tmp_path: Path) -> None:
    """load_custom_graders returns empty Graders when agent.toml has no [graders]."""
    assert load_custom_graders(agent_config(tmp_path)) == Graders()


BAD_JUDGES: dict[str, tuple[dict[str, Any], str]] = {
    "placeholder": (
        {"prompt": "openevals:RAG_GROUNDEDNESS_PROMPT"},
        re.escape(
            "[judges.j] prompt may use only ['inputs', 'outputs', "
            "'reference_outputs'], but uses ['context', 'outputs']"
        ),
    ),
    "unknown_openevals_prompt": (
        {"prompt": "openevals:__name__"},
        re.escape("[judges.j] openevals.prompts has no prompt '__name__'"),
    ),
    "reference_free_with_reference": (
        {"prompt": "openevals:CORRECTNESS_PROMPT", "reference_free": True},
        re.escape("[judges.j] a reference_free judge's prompt cannot use {reference_"),
    ),
    "continuous_gate": (
        {"prompt": "x.md", "continuous": True, "gate": True},
        re.escape("[judges] j: Value error, a gate judge must not be continuous"),
    ),
    "bad_model": (
        {"prompt": "x.md", "model": "gpt-5.5"},
        re.escape("[judges] j.model: String should match pattern"),
    ),
    "unknown_key": (
        {"prompt": "x.md", "temperature": 0},
        re.escape("[judges] j.temperature: Extra inputs are not permitted"),
    ),
    "missing_file": ({"prompt": "missing.md"}, r"file not found: .*missing\.md"),
}


@pytest.mark.parametrize(
    ("judge", "reason"), BAD_JUDGES.values(), ids=BAD_JUDGES.keys()
)
def test_bad_judge_table_fails_at_load_naming_it(
    tmp_path: Path, judge: dict[str, Any], reason: str
) -> None:
    """A judge code cannot fill or run raises AgentConfigError before any case."""
    config = agent_config(tmp_path, judges={"j": judge})
    with pytest.raises(AgentConfigError, match=reason):
        build_graders(config, model_factory=ModelFactory())


def test_a_prompt_file_with_a_stray_brace_or_a_taken_name_fails(tmp_path: Path) -> None:
    """A prompt str.format cannot parse, or a built-in key as a name, is rejected."""
    (tmp_path / "judge.md").write_text("Grade {outputs} as JSON {", encoding="utf-8")
    stray = agent_config(tmp_path, judges={"j": {"prompt": "judge.md"}})
    taken = agent_config(tmp_path, judges={"criteria": {"prompt": "judge.md"}})

    with pytest.raises(AgentConfigError, match=r"\[judges\.j\] Single '\{'"):
        build_graders(stray, model_factory=ModelFactory())
    with pytest.raises(AgentConfigError, match="'criteria' is a built-in grader's"):
        build_graders(taken, model_factory=ModelFactory())


@pytest.mark.parametrize("name", ["pass_rate", "error_rate", "rows"])
def test_a_judge_cannot_take_a_key_summarize_reports(tmp_path: Path, name: str) -> None:
    """A judge named like a summary key fails at load, not in summarize after a run."""
    config = agent_config(tmp_path, judges={name: {"prompt": "openevals:X"}})

    with pytest.raises(AgentConfigError, match=f"'{name}' is .* a summary key"):
        build_graders(config, model_factory=ModelFactory())


BAD_FACTORIES = {
    "not_module_attr": ("tests.golden", "factory: String should match pattern"),
    "missing_module": ("tests.golden.no_such_module:build", "No module named"),
    "missing_attr": ("tests.golden.test_golden_graders:nope", "nope"),
    "not_graders": (
        "tests.golden.test_golden_graders:_not_graders",
        "returned something other than Graders",
    ),
}


@pytest.mark.parametrize(
    ("factory", "reason"), BAD_FACTORIES.values(), ids=BAD_FACTORIES.keys()
)
def test_bad_graders_factory_fails_naming_the_table(
    tmp_path: Path, factory: str, reason: str
) -> None:
    """A factory that cannot be imported or returns no Graders is a config error."""
    config = agent_config(tmp_path, graders={"factory": factory})
    with pytest.raises(AgentConfigError, match=rf"agent\.toml: \[graders\] .*{reason}"):
        load_custom_graders(config)
