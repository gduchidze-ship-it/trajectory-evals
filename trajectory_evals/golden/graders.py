"""Graders of golden cases: deterministic checks, the criteria judge, agent judges.

Each grader returns unscored, with a short comment, when its expectation is
missing or empty, when the target errored, and, for tools_called and
trajectory_match, when the agent reported no trajectory. A failing grader's
comment names what is missing; a reflection model reads it.
"""

import json
import logging
import pkgutil
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, Self, cast

import openevals.prompts
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from openevals import create_async_llm_as_judge, create_async_trajectory_match_evaluator
from openevals.utils import _chat_completion_messages_to_string
from pydantic import StringConstraints, TypeAdapter, ValidationError, model_validator

from trajectory_evals.core.agent_config import (
    AGENT_FILE,
    AgentConfig,
    AgentConfigError,
    ImportPath,
)
from trajectory_evals.core.evaluators import _async_evaluator, _target_errored
from trajectory_evals.core.jsonl import _describe
from trajectory_evals.core.settings import Text, _placeholders, _Section
from trajectory_evals.core.types import (
    AgentOutputs,
    ChatCompletionMessage,
    EvaluatorResult,
    Expectations,
    SimpleAsyncEvaluator,
    SimpleEvaluator,
    ToolArgsMatch,
    TrajectoryMatch,
)
from trajectory_evals.golden.settings import settings

logger = logging.getLogger(__name__)

type ModelFactory = Callable[[str], BaseChatModel]
type Score = bool | tuple[bool | float | None, str | None]
type Verdict = Callable[..., Awaitable[Score]]

REFERENCE_PLACEHOLDER = "reference_outputs"
JUDGE_PLACEHOLDERS = frozenset({"inputs", "outputs", REFERENCE_PLACEHOLDER})
DEFAULT_TRAJECTORY_MATCH: TrajectoryMatch = "strict"
DEFAULT_TOOL_ARGS_MATCH: ToolArgsMatch = "exact"

ModelName = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^\w+:\S+$")
]

_ERRORED: Score = (None, settings.comments.target_errored)
_UNREPORTED: Score = (None, settings.comments.trajectory_unreported)
_JUDGE_ERROR: Score = (None, settings.comments.judge_error)


class JudgeSection(_Section):
    """[judges.<name>]: a judge of every row's full transcript, keyed by its name.

    prompt is a file in the agent folder, or "openevals:<NAME>" for a prompt in
    openevals.prompts. It may use only {inputs} (the case inputs as JSON),
    {outputs} (the full transcript) and {reference_outputs} (the expected dict as
    JSON); a reference_free judge's prompt cannot use {reference_outputs}. model
    is a provider:model string, defaulting to the golden [judges].model. Only a
    gate judge, which must be binary, joins the default pass_keys.
    """

    prompt: Text
    model: ModelName | None = None
    continuous: bool = False
    reference_free: bool = False
    gate: bool = False

    @model_validator(mode="after")
    def _check_gate(self) -> Self:
        """Reject a continuous gate, which pass_rate cannot AND."""
        if self.gate and self.continuous:
            raise ValueError(settings.texts.continuous_gate)
        return self


class GradersSection(_Section):
    """[graders]: the module:attr factory(config) -> Graders an agent adds."""

    factory: ImportPath


@dataclass(frozen=True)
class Graders:
    """Row evaluators, summary evaluators, and the feedback keys pass_rate ANDs."""

    evaluators: tuple[SimpleAsyncEvaluator | SimpleEvaluator, ...] = ()
    summary_evaluators: tuple[Callable[..., Any], ...] = ()
    pass_keys: tuple[str, ...] = ()


_JUDGES = TypeAdapter(dict[str, JudgeSection])
_GRADERS = TypeAdapter(GradersSection)


def _evaluator(scorer: Callable[..., Any], feedback_key: str) -> SimpleAsyncEvaluator:
    """Wrap scorer as an async evaluator under feedback_key that also gets inputs.

    aevaluate passes only the arguments an evaluator's signature names, and
    core's wrapper does not name inputs, which the graded span and judges read.
    """
    evaluator = _async_evaluator(
        scorer, feedback_key, ls_framework=settings.langsmith.ls_framework
    )

    async def wrapper(
        *,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any],
        reference_outputs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> EvaluatorResult | list[EvaluatorResult]:
        """Score one run under the feedback key."""
        return await evaluator(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            **kwargs,
        )

    wrapper.__name__ = feedback_key
    return wrapper


def _expected(reference_outputs: Expectations | None) -> Expectations:
    """Return the case's expectations, none when there is no reference."""
    return reference_outputs or Expectations()


def _not_expected(key: str) -> Score:
    """Return the unscored result of a grader whose expectation is absent."""
    return None, settings.comments.not_expected.format(key=key)


def _verdict(failures: list[str], template: str) -> Score:
    """Pass with no failures; else fail with template naming them."""
    if not failures:
        return True
    items = settings.comments.item_separator.join(failures)
    return False, template.format(items=items)


def _span(
    inputs: Mapping[str, Any] | None, outputs: AgentOutputs
) -> list[ChatCompletionMessage]:
    """Return the graded span: from the last input message on, or everything."""
    sent = (inputs or {}).get("messages") or ()
    return outputs["messages"][max(len(sent) - 1, 0) :]


def _replies(inputs: Mapping[str, Any] | None, outputs: AgentOutputs) -> str:
    """Return the casefolded text of every assistant message in the graded span."""
    assistant = settings.data.assistant_role
    return "\n".join(
        _text(message.get("content")).casefold()
        for message in _span(inputs, outputs)
        if message["role"] == assistant
    )


def _text(content: str | list[dict[str, Any]] | None) -> str:
    """Return a message content's text, joining the text of a list of parts."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or ())


def _trajectory_reported(outputs: AgentOutputs) -> bool:
    """Return whether the agent reported its tool calls; a missing key means it did."""
    return outputs.get("trajectory_reported", True)


def _score_output(
    *,
    outputs: AgentOutputs,
    reference_outputs: Expectations | None = None,
    **_: Any,
) -> Score:
    """Pass when the final reply equals expected.output, ignoring outer whitespace."""
    if _target_errored(outputs):
        return _ERRORED
    expected = _expected(reference_outputs).get("output")
    if not expected:
        return _not_expected("output")
    if outputs["text"].strip() == expected.strip():
        return True
    return False, settings.comments.output_mismatch.format(expected=expected.strip())


def _score_contains(
    *,
    inputs: Mapping[str, Any] | None = None,
    outputs: AgentOutputs,
    reference_outputs: Expectations | None = None,
    **_: Any,
) -> Score:
    """Pass when the span's assistant replies hold every expected.contains phrase."""
    if _target_errored(outputs):
        return _ERRORED
    phrases = _expected(reference_outputs).get("contains")
    if not phrases:
        return _not_expected("contains")
    replies = _replies(inputs, outputs)
    missing = [phrase for phrase in phrases if phrase.casefold() not in replies]
    return _verdict(missing, settings.comments.missing)


def _score_not_contains(
    *,
    inputs: Mapping[str, Any] | None = None,
    outputs: AgentOutputs,
    reference_outputs: Expectations | None = None,
    **_: Any,
) -> Score:
    """Pass when no assistant reply in the span holds an expected.not_contains one."""
    if _target_errored(outputs):
        return _ERRORED
    phrases = _expected(reference_outputs).get("not_contains")
    if not phrases:
        return _not_expected("not_contains")
    replies = _replies(inputs, outputs)
    found = [phrase for phrase in phrases if phrase.casefold() in replies]
    return _verdict(found, settings.comments.forbidden)


def _score_tools(
    *,
    inputs: Mapping[str, Any] | None = None,
    outputs: AgentOutputs,
    reference_outputs: Expectations | None = None,
    **_: Any,
) -> Score:
    """Pass when the span calls every expected.tools name at least once."""
    if _target_errored(outputs):
        return _ERRORED
    tools = _expected(reference_outputs).get("tools")
    if not tools:
        return _not_expected("tools")
    if not _trajectory_reported(outputs):
        return _UNREPORTED
    called = {
        call["function"]["name"]
        for message in _span(inputs, outputs)
        for call in message.get("tool_calls") or ()
    }
    missing = [tool for tool in tools if tool not in called]
    return _verdict(missing, settings.comments.tools_missing)


@cache
def _trajectory_evaluator(
    match: TrajectoryMatch, tool_args: ToolArgsMatch
) -> SimpleAsyncEvaluator:
    """Return openevals' trajectory match for one pair of modes, made once."""
    return create_async_trajectory_match_evaluator(
        trajectory_match_mode=match, tool_args_match_mode=tool_args
    )


async def _score_trajectory(
    *,
    inputs: Mapping[str, Any] | None = None,
    outputs: AgentOutputs,
    reference_outputs: Expectations | None = None,
    **_: Any,
) -> Score:
    """Pass when the span matches expected.trajectory in the case's modes."""
    if _target_errored(outputs):
        return _ERRORED
    expected = _expected(reference_outputs)
    reference = expected.get("trajectory")
    if not reference:
        return _not_expected("trajectory")
    if not _trajectory_reported(outputs):
        return _UNREPORTED
    match = expected.get("trajectory_match", DEFAULT_TRAJECTORY_MATCH)
    tool_args = expected.get("tool_args_match", DEFAULT_TOOL_ARGS_MATCH)
    # openevals normalizes messages in place, so it gets copies.
    result = await _trajectory_evaluator(match, tool_args)(
        outputs=[dict(message) for message in _span(inputs, outputs)],
        reference_outputs=[dict(message) for message in reference],
    )
    if cast(EvaluatorResult, result)["score"]:
        return True
    comment = settings.comments.trajectory_mismatch
    return False, comment.format(trajectory_match=match, tool_args_match=tool_args)


_keys = settings.keys
output_exact = _evaluator(_score_output, _keys.output_exact)
contains = _evaluator(_score_contains, _keys.contains)
not_contains = _evaluator(_score_not_contains, _keys.not_contains)
tools_called = _evaluator(_score_tools, _keys.tools_called)
trajectory_match = _evaluator(_score_trajectory, _keys.trajectory_match)

_DETERMINISTIC: Mapping[str, SimpleAsyncEvaluator] = {
    _keys.output_exact: output_exact,
    _keys.contains: contains,
    _keys.not_contains: not_contains,
    _keys.tools_called: tools_called,
    _keys.trajectory_match: trajectory_match,
}


def _judge(judge: Callable[..., Awaitable[Any]], feedback_key: str) -> Verdict:
    """Return a call of a built openevals judge giving its score and reasoning.

    Only calling the model and reading its reply happen here, so only their
    failures become unscored judge_error; making the model raised already.
    """

    async def verdict(**arguments: Any) -> Score:
        """Return the judge's score and reasoning; unscored if the judge fails."""
        try:
            result = await judge(**arguments)
            return result["score"], result["comment"]
        except Exception as e:
            logger.warning(settings.texts.judge_failed, feedback_key, type(e).__name__)
            return _JUDGE_ERROR

    return verdict


def _escape_tags(data: str) -> str:
    """Return data with every closing tag escaped, so it cannot end its own tag."""
    return data.replace("</", "<\\/")


def _transcript(outputs: AgentOutputs) -> str:
    """Return the full transcript as the text openevals' trajectory judges read."""
    return _chat_completion_messages_to_string(outputs["messages"])


def _criteria(model: str, model_factory: ModelFactory) -> SimpleAsyncEvaluator:
    """Return the evaluator whose one judge grades all of a case's criteria lines.

    The grading rules go in the system message and only data in the user
    message, with closing tags inside the data escaped, so text in a reply,
    tool result or context cannot step outside its tag to instruct the judge.
    """
    key, judges = settings.keys.criteria, settings.judges
    judge = create_async_llm_as_judge(
        prompt=judges.criteria_prompt,
        system=judges.criteria_system,
        judge=model_factory(model),
        feedback_key=key,
    )
    verdict = _judge(judge, key)

    async def score(
        *,
        inputs: Mapping[str, Any] | None = None,
        outputs: AgentOutputs,
        reference_outputs: Expectations | None = None,
        **_: Any,
    ) -> Score:
        """Pass when the judge finds every expected.criteria line holds."""
        if _target_errored(outputs):
            return _ERRORED
        criteria = _expected(reference_outputs).get("criteria")
        if not criteria:
            return _not_expected("criteria")
        context = (inputs or {}).get("context", {})
        return await verdict(
            context=_escape_tags(json.dumps(context, ensure_ascii=False)),
            transcript=_escape_tags(_transcript(outputs)),
            criteria=_escape_tags(json.dumps(criteria, ensure_ascii=False)),
        )

    return _evaluator(score, key)


def _agent_judge(
    config: AgentConfig, name: str, section: JudgeSection, model_factory: ModelFactory
) -> SimpleAsyncEvaluator:
    """Return the evaluator of one [judges.<name>] table, reported as name."""
    judge = create_async_llm_as_judge(
        prompt=_judge_prompt(config, name, section),
        judge=model_factory(section.model or settings.judges.model),
        feedback_key=name,
        continuous=section.continuous,
    )
    verdict = _judge(judge, name)

    async def score(
        *,
        inputs: Mapping[str, Any] | None = None,
        outputs: AgentOutputs,
        reference_outputs: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Score:
        """Return the judge's verdict on the row's full transcript."""
        if _target_errored(outputs):
            return _ERRORED
        # openevals rewrites a dict's messages key in place, so it gets copies.
        return await verdict(
            inputs=dict(inputs or {}),
            outputs=_transcript(outputs),
            reference_outputs=dict(reference_outputs or {}),
        )

    return _evaluator(score, name)


def _judge_prompt(config: AgentConfig, name: str, section: JudgeSection) -> str:
    """Return the judge's prompt, raising AgentConfigError if code cannot fill it."""
    table, texts = f"judges.{name}", settings.texts
    if name in _taken_keys():
        raise _table_error(config, table, texts.key_taken.format(key=name))
    prompt = _prompt_text(config, table, section.prompt)
    try:
        used = _placeholders(prompt)
    except ValueError as error:
        raise _table_error(config, table, str(error)) from None
    if not used <= JUDGE_PLACEHOLDERS:
        allowed, named = sorted(JUDGE_PLACEHOLDERS), sorted(used)
        message = texts.bad_placeholders.format(allowed=allowed, used=named)
        raise _table_error(config, table, message)
    if section.reference_free and REFERENCE_PLACEHOLDER in used:
        raise _table_error(config, table, texts.reference_free_prompt)
    return prompt


def _prompt_text(config: AgentConfig, table: str, prompt: str) -> str:
    """Return an openevals prompt by "openevals:<NAME>", else the folder's file."""
    prefix = settings.judges.openevals_prefix
    if not prompt.startswith(prefix):
        return config.read_text(prompt)
    name = prompt.removeprefix(prefix)
    if name not in openevals.prompts.__all__:
        message = settings.texts.unknown_prompt.format(name=name)
        raise _table_error(config, table, message)
    return cast(str, getattr(openevals.prompts, name))


def _taken_keys() -> set[str]:
    """Return the built-in graders' feedback keys, which a judge cannot reuse."""
    return set(settings.keys.model_dump().values())


def _table[Table](config: AgentConfig, name: str, adapter: TypeAdapter[Table]) -> Table:
    """Validate the named agent.toml table, raising AgentConfigError naming a field."""
    try:
        return adapter.validate_python(config.section(name))
    except ValidationError as error:
        raise _table_error(config, name, _describe(error)) from None


def _table_error(config: AgentConfig, table: str, message: str) -> AgentConfigError:
    """Return the error for a table of the agent's agent.toml."""
    path = config.root / AGENT_FILE
    texts = settings.texts
    return AgentConfigError(
        texts.table_error.format(path=path, table=table, message=message)
    )


def load_custom_graders(config: AgentConfig) -> Graders:
    """Return the Graders [graders].factory(config) builds; none without the table.

    Raises AgentConfigError when the factory cannot be imported or returns
    something other than Graders; an error the factory itself raises propagates.
    """
    if not config.section("graders"):
        return Graders()
    factory_path = _table(config, "graders", _GRADERS).factory
    try:
        factory = pkgutil.resolve_name(factory_path)
    except (ImportError, AttributeError, ValueError) as error:
        raise _table_error(config, "graders", str(error)) from error
    graders = factory(config)
    if not isinstance(graders, Graders):
        message = settings.texts.not_graders.format(factory=factory_path)
        raise _table_error(config, "graders", message)
    return graders


def build_graders(
    config: AgentConfig,
    *,
    model_factory: ModelFactory = init_chat_model,
    criteria: bool = True,
    criteria_model: str | None = None,
) -> Graders:
    """Return the graders of an agent's golden runs, in report order.

    These are the deterministic graders, the criteria judge, each [judges.<name>]
    and the [graders] factory's own. criteria=False leaves the criteria judge
    and its pass key out and makes no model for it, so a dataset without
    expected.criteria needs no LLM API key; the caller decides from the cases.
    Every judge's chat model is model_factory(provider:model), made here:
    criteria_model for the criteria judge, a judge's own model for an agent
    judge, else the golden [judges].model. pass_keys are the deterministic keys,
    criteria, each gate judge's name and the factory's pass_keys; a judge pass
    never overrides a deterministic fail, because pass_rate ANDs them.

    Raises AgentConfigError for a bad [judges] or [graders] table; an error
    model_factory raises, such as a missing API key or provider package,
    propagates rather than leaving every row unjudged.
    """
    judges = _table(config, "judges", _JUDGES)
    agent_judges = [
        _agent_judge(config, name, section, model_factory)
        for name, section in judges.items()
    ]
    custom = load_custom_graders(config)
    built_in = dict(_DETERMINISTIC)
    if criteria:
        model = criteria_model or settings.judges.model
        built_in[settings.keys.criteria] = _criteria(model, model_factory)
    gates = [name for name, section in judges.items() if section.gate]
    return Graders(
        evaluators=(*built_in.values(), *agent_judges, *custom.evaluators),
        summary_evaluators=custom.summary_evaluators,
        pass_keys=(*built_in, *gates, *custom.pass_keys),
    )


__all__ = [
    "DEFAULT_TOOL_ARGS_MATCH",
    "DEFAULT_TRAJECTORY_MATCH",
    "JUDGE_PLACEHOLDERS",
    "Graders",
    "GradersSection",
    "JudgeSection",
    "ModelFactory",
    "build_graders",
    "contains",
    "load_custom_graders",
    "not_contains",
    "output_exact",
    "tools_called",
    "trajectory_match",
]
