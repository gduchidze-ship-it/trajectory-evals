"""Tests for scoring golden runs with the deterministic graders and the judges."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import JUDGE_REASONING, FailingChatModel, judge_reply, scripted
from tests.golden.golden_fakes import TOOL_CALL, ModelFactory, agent_config
from trajectory_evals.core.types import (
    AgentOutputs,
    ChatCompletionMessage,
    SimpleAsyncEvaluator,
    ToolMessage,
)
from trajectory_evals.golden.graders import (
    Graders,
    build_graders,
    contains,
    not_contains,
    output_exact,
    tools_called,
    trajectory_match,
)

ASKED: ChatCompletionMessage = {"role": "user", "content": "Do you price match?"}
ANSWERED: ChatCompletionMessage = {
    "role": "assistant",
    "content": "Yes, we price match.",
}
USER: ChatCompletionMessage = {"role": "user", "content": "Refund order A1."}
CALL: ChatCompletionMessage = {
    "role": "assistant",
    "content": "Checking A1 now.",
    "tool_calls": [TOOL_CALL],
}
RESULT = ToolMessage(role="tool", content="refunded", tool_call_id="call-1")
REPLY: ChatCompletionMessage = {
    "role": "assistant",
    "content": "Your Refund is on its way.",
}
INPUTS = {"messages": [ASKED, ANSWERED, USER], "context": {"channel": "sms"}}
OUTPUTS = AgentOutputs(
    text="Your Refund is on its way.",
    messages=[ASKED, ANSWERED, USER, CALL, RESULT, REPLY],
)
ERRORED = {"output": None}


def _result(key: str, score: Any, comment: str | None = None) -> dict[str, Any]:
    """Return the EvaluatorResult a grader gives."""
    return {"key": key, "score": score, "comment": comment, "metadata": None}


async def _score(
    evaluator: SimpleAsyncEvaluator,
    expected: dict[str, Any],
    outputs: Any = OUTPUTS,
    inputs: Any = INPUTS,
) -> Any:
    """Score one run as aevaluate would, by keyword."""
    return await evaluator(inputs=inputs, outputs=outputs, reference_outputs=expected)


GRADED: dict[str, tuple[SimpleAsyncEvaluator, dict[str, Any], Any, str | None]] = {
    "output_pass": (
        output_exact,
        {"output": " Your Refund is on its way.\n"},
        True,
        None,
    ),
    "output_fail": (
        output_exact,
        {"output": "Refunded."},
        False,
        "expected exactly: Refunded.",
    ),
    "contains_pass": (contains, {"contains": ["REFUND", "on its way"]}, True, None),
    "contains_fail": (
        contains,
        {"contains": ["refund", "store credit", "receipt"]},
        False,
        "missing: store credit, receipt",
    ),
    "not_contains_pass": (not_contains, {"not_contains": ["guarantee"]}, True, None),
    "not_contains_fail": (
        not_contains,
        {"not_contains": ["guarantee", "ON ITS WAY"]},
        False,
        "forbidden: ON ITS WAY",
    ),
    "tools_pass": (tools_called, {"tools": ["lookup_order"]}, True, None),
    "tools_fail": (
        tools_called,
        {"tools": ["lookup_order", "refund_order"]},
        False,
        "tools not called: refund_order",
    ),
}


@pytest.mark.parametrize(
    ("evaluator", "expected", "score", "comment"), GRADED.values(), ids=GRADED.keys()
)
@pytest.mark.asyncio
async def test_deterministic_grader_passes_or_names_what_is_missing(
    evaluator: SimpleAsyncEvaluator,
    expected: dict[str, Any],
    score: bool,
    comment: str | None,
) -> None:
    """Each grader scores its own expectation; a failure's comment names the gap."""
    key = getattr(evaluator, "__name__")
    assert await _score(evaluator, expected) == _result(key, score, comment)


GRADERS = {
    "output": output_exact,
    "contains": contains,
    "not_contains": not_contains,
    "tools": tools_called,
    "trajectory": trajectory_match,
}


@pytest.mark.parametrize(("name", "evaluator"), GRADERS.items(), ids=GRADERS.keys())
@pytest.mark.asyncio
async def test_grader_is_unscored_without_its_expectation_or_a_run(
    name: str, evaluator: SimpleAsyncEvaluator
) -> None:
    """No expected key, or an empty one, or an errored target: unscored, with why."""
    key = getattr(evaluator, "__name__")
    absent = _result(key, None, f"no expected.{name}")

    assert await _score(evaluator, {"action": "BLOCKED"}) == absent
    assert await _score(evaluator, {name: []}) == absent
    assert await _score(evaluator, {name: ["x"]}, outputs=ERRORED) == _result(
        key, None, "target errored"
    )


@pytest.mark.asyncio
async def test_text_graders_read_every_assistant_reply_after_the_history() -> None:
    """History and tool results are not graded; each reply in the span is."""
    assert (await _score(contains, {"contains": ["price match"]}))["score"] is False
    assert (await _score(contains, {"contains": ["refunded"]}))["score"] is False
    assert (await _score(contains, {"contains": ["checking a1"]}))["score"] is True
    assert (await _score(not_contains, {"not_contains": ["price match"]}))[
        "score"
    ] is True
    assert (await _score(not_contains, {"not_contains": ["checking"]}))[
        "score"
    ] is False


@pytest.mark.asyncio
async def test_without_input_messages_the_whole_conversation_is_graded() -> None:
    """A simulation's inputs carry no messages, so history counts too."""
    expected = {"contains": ["price match"]}
    for inputs in (None, {"persona": "frustrated"}):
        assert (await _score(contains, expected, inputs=inputs))["score"] is True


@pytest.mark.parametrize("evaluator", [tools_called, trajectory_match])
@pytest.mark.asyncio
async def test_trajectory_graders_are_unscored_when_none_was_reported(
    evaluator: SimpleAsyncEvaluator,
) -> None:
    """An agent that sent no trajectory is not failed for calling no tools."""
    outputs = AgentOutputs(
        text="Your Refund is on its way.",
        messages=[USER, REPLY],
        trajectory_reported=False,
    )
    expected = {"tools": ["lookup_order"], "trajectory": [USER, REPLY]}
    assert await _score(evaluator, expected, outputs=outputs) == _result(
        getattr(evaluator, "__name__"), None, "the agent reported no trajectory"
    )


def _call(name: str, **arguments: str) -> dict[str, Any]:
    """Return an OpenAI-format tool call of name with arguments."""
    function = {"name": name, "arguments": json.dumps(arguments)}
    return {"id": f"call-{name}", "type": "function", "function": function}


@pytest.mark.asyncio
async def test_strict_trajectory_matches_the_span_turn_by_turn() -> None:
    """Strict compares the span from the last user message; args must be exact."""
    reference = [USER, CALL, RESULT, {"role": "assistant", "content": "Done."}]
    outputs = copy.deepcopy(OUTPUTS)
    other_order = {**CALL, "tool_calls": [_call("lookup_order", order_id="B2")]}

    assert await _score(trajectory_match, {"trajectory": reference}) == _result(
        "trajectory_match", True
    )
    assert await _score(
        trajectory_match, {"trajectory": [USER, other_order, RESULT, REPLY]}
    ) == _result(
        "trajectory_match",
        False,
        "trajectory does not match the reference (strict trajectory, exact tool args)",
    )
    assert outputs == OUTPUTS


@pytest.mark.asyncio
async def test_superset_trajectory_allows_extra_calls() -> None:
    """Superset passes when every reference call is made, among others."""
    both: ChatCompletionMessage = {
        "role": "assistant",
        "content": "",
        "tool_calls": [_call("lookup_order", order_id="A1"), _call("refund_order")],
    }
    inputs = {"messages": [USER], "context": {}}
    outputs = AgentOutputs(text="Done.", messages=[USER, both, REPLY])
    reference = [
        {"role": "assistant", "content": "", "tool_calls": [_call("refund_order")]}
    ]
    strict: dict[str, Any] = {"trajectory": reference}
    superset = {**strict, "trajectory_match": "superset"}

    assert (await _score(trajectory_match, strict, outputs, inputs))["score"] is False
    assert (await _score(trajectory_match, superset, outputs, inputs))["score"] is True


def _graders(tmp_path: Path, factory: ModelFactory, **sections: Any) -> dict[str, Any]:
    """Build the agent's graders with factory and return them by feedback key."""
    graders: Graders = build_graders(
        agent_config(tmp_path, **sections), model_factory=factory
    )
    return {
        getattr(evaluator, "__name__"): evaluator for evaluator in graders.evaluators
    }


@pytest.mark.parametrize("verdict", [True, False])
@pytest.mark.asyncio
async def test_criteria_judge_grades_every_line_in_one_call(
    tmp_path: Path, verdict: bool
) -> None:
    """One judge call sees the context, the transcript and every criterion."""
    judge = scripted(judge_reply(verdict))
    factory = ModelFactory(judge)
    criteria = _graders(tmp_path, factory)["criteria"]
    expected = {"criteria": ["Confirms the refund", "Stays under 160 characters"]}

    assert await _score(criteria, expected) == _result(
        "criteria", verdict, JUDGE_REASONING
    )
    prompt = judge.prompts[0][0].text
    assert '["Confirms the refund", "Stays under 160 characters"]' in prompt
    assert '{"channel": "sms"}' in prompt
    assert "<user>\nDo you price match?\n</user>" in prompt
    assert "<tool_result>" in prompt
    assert factory.names == ["openai:gpt-5.5"]


@pytest.mark.parametrize(
    "judge", [FailingChatModel(messages=iter(())), scripted("no tool call")]
)
@pytest.mark.asyncio
async def test_a_failing_or_malformed_judge_is_unscored(
    tmp_path: Path, judge: Any
) -> None:
    """A judge that raises or replies without a score is judge_error, not a crash."""
    criteria = _graders(tmp_path, ModelFactory(judge))["criteria"]
    assert await _score(criteria, {"criteria": ["Confirms the refund"]}) == _result(
        "criteria", None, "judge_error"
    )


@pytest.mark.asyncio
async def test_criteria_without_lines_or_a_run_never_makes_the_model(
    tmp_path: Path,
) -> None:
    """The judge's model is made on first use, so unjudged rows need no API key."""
    factory = ModelFactory()
    criteria = _graders(tmp_path, factory)["criteria"]

    assert await _score(criteria, {}) == _result(
        "criteria", None, "no expected.criteria"
    )
    assert await _score(criteria, {"criteria": ["x"]}, outputs=ERRORED) == _result(
        "criteria", None, "target errored"
    )
    assert factory.names == []


@pytest.mark.asyncio
async def test_openevals_prompt_judges_the_full_transcript(tmp_path: Path) -> None:
    """An openevals: prompt resolves, and the judge reports under its own name."""
    judge = scripted(judge_reply(True))
    factory = ModelFactory(judge)
    section = {
        "prompt": "openevals:TASK_COMPLETION_PROMPT",
        "model": "bedrock_converse:x",
    }
    done = _graders(tmp_path, factory, judges={"done": section})["done"]

    assert await _score(done, {}) == _result("done", True, JUDGE_REASONING)
    assert "<user>\nDo you price match?\n</user>" in judge.prompts[0][0].text
    assert factory.names == ["bedrock_converse:x"]


@pytest.mark.asyncio
async def test_file_prompt_judge_reads_inputs_outputs_and_reference(
    tmp_path: Path,
) -> None:
    """A prompt file's three placeholders are the inputs, transcript and expected."""
    (tmp_path / "judge.md").write_text(
        "IN {inputs}\nOUT {outputs}\nREF {reference_outputs}", encoding="utf-8"
    )
    judge = scripted(judge_reply(False))
    polite = _graders(
        tmp_path, ModelFactory(judge), judges={"polite": {"prompt": "judge.md"}}
    )["polite"]

    assert await _score(polite, {"action": "REFUND"}) == _result(
        "polite", False, JUDGE_REASONING
    )
    prompt = judge.prompts[0][0].text
    assert '"context": {"channel": "sms"}' in prompt
    assert "OUT <user>\nDo you price match?\n</user>" in prompt
    assert 'REF {"action": "REFUND"}' in prompt
