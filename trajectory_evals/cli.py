"""Command line and composition root: check, serve, eval, simulate and redteam."""

import argparse
import asyncio
import ipaddress
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import uvicorn
from a2a.utils.errors import A2AError
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langsmith import Client
from pydantic import TypeAdapter, ValidationError
from starlette.applications import Starlette

from trajectory_evals.cli_settings import settings
from trajectory_evals.connect.bridge import app_from_config
from trajectory_evals.connect.client import A2AAgent
from trajectory_evals.core.agent_config import (
    AgentConfig,
    AgentConfigError,
    load_agent,
)
from trajectory_evals.core.artifacts import write_run
from trajectory_evals.core.dataset import LangSmithDatasetSync
from trajectory_evals.core.jsonl import CaseFormatError
from trajectory_evals.core.protocols import CaseSource
from trajectory_evals.core.runner import (
    ExperimentConfig,
    ExperimentRunner,
    Target,
    build_local_client,
)
from trajectory_evals.core.settings import settings as core_settings
from trajectory_evals.core.summary import summarize
from trajectory_evals.core.types import AgentError, DatasetExample
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.golden.graders import (
    Graders,
    ModelFactory,
    ModelName,
    build_graders,
)
from trajectory_evals.golden.target import GoldenTarget
from trajectory_evals.simulation import (
    RedteamSection,
    ScenarioSource,
    SimulationTarget,
    agent_table,
    create_async_attack_succeeded_evaluator,
    create_async_turns_to_break_evaluator,
    load_personas,
    redteam_cases,
)
from trajectory_evals.simulation.settings import settings as simulation_settings

_LOCALHOST = "localhost"
_DATASET_PREFIX_KEY = "dataset_prefix"  # the [langsmith] key naming the datasets
_MODEL_NAME = TypeAdapter(ModelName)


class PreflightError(Exception):
    """Bad input found before a run: a missing key, file or case, or a bad name."""


# What every command reports as bad input: a missing prerequisite, a bad agent
# folder or data row, or an agent that fails, such as an unset token variable.
_INPUT_ERRORS = (PreflightError, AgentConfigError, CaseFormatError, AgentError)
# check also reports a module:attr that does not resolve, bad [bridge] options
# and an agent card it cannot fetch, since finding those is its job.
_CHECK_ERRORS = (*_INPUT_ERRORS, ValueError, ImportError, AttributeError, A2AError)


@dataclass(frozen=True)
class Builders:
    """Factories for the LangSmith clients, the agent, chat models and the server."""

    client: Callable[[], Client]
    local_client: Callable[[], Client]
    agent: Callable[[AgentConfig], A2AAgent]
    model_factory: ModelFactory
    serve: Callable[[Starlette, str, int], None]


def _serve(app: Starlette, host: str, port: int) -> None:
    """Serve app over HTTP on host:port until the process is interrupted."""
    uvicorn.run(app, host=host, port=port)


DEFAULT_BUILDERS = Builders(
    client=Client,
    local_client=build_local_client,
    agent=A2AAgent.from_config,
    model_factory=init_chat_model,
    serve=_serve,
)


def _say(text: str) -> None:
    """Write one line of the command's own output to stdout."""
    sys.stdout.write(f"{text}\n")


def _warn(text: str) -> None:
    """Write one warning or error line to stderr."""
    sys.stderr.write(f"{text}\n")


def _preflight(args: argparse.Namespace) -> None:
    """Raise PreflightError for a missing LangSmith key or data file."""
    texts, env = settings.texts, settings.cli.langsmith_api_key_env
    # check and serve upload nothing, so they have no --local.
    uploads = not getattr(args, "local", True)
    if uploads and not os.environ.get(env):
        raise PreflightError(texts.no_langsmith_key.format(env=env))
    if args.data is not None and not args.data.is_file():
        raise PreflightError(texts.data_not_found.format(path=args.data))


def _model_stand_in(name: str) -> BaseChatModel:
    """Return a model check never calls, once name is a provider:model string."""
    try:
        _MODEL_NAME.validate_python(name)
    except ValidationError:
        raise AgentConfigError(settings.texts.bad_model.format(name=name)) from None
    # check validates config only, so no API key or provider package is needed.
    return GenericFakeChatModel(messages=iter(()))


def _source(path: Path) -> CaseSource:
    """Return path's cases: scenarios if its name says so, else golden cases."""
    if settings.check.scenario_marker in path.name:
        return ScenarioSource(path)
    return JsonlGoldenSource(path)


def _dataset_counts(config: AgentConfig) -> list[tuple[Path, int]]:
    """Return each datasets file of the folder, relative, with its case count."""
    check = settings.check
    paths = sorted((config.root / check.datasets_dir).glob(check.datasets_glob))
    return [
        (path.relative_to(config.root), sum(1 for _ in _source(path).examples()))
        for path in paths
    ]


async def run_check(args: argparse.Namespace, builders: Builders) -> None:
    """Validate the agent's tables, datasets and card; with --ping, send a message.

    Every table is validated as eval, simulate and redteam read it, with model
    names checked as provider:model strings and no model built.
    """
    texts = settings.texts
    config = load_agent(args.agent)
    _say(texts.agent_line.format(name=config.name, root=config.root))
    build_graders(config, model_factory=_model_stand_in, criteria=False)
    for path, count in _dataset_counts(config):
        _say(texts.dataset_line.format(path=path, count=count))
    async with builders.agent(config) as agent:
        SimulationTarget(agent, config=config, model_factory=_model_stand_in)
        _say(texts.tables_line)
        # shortcut: A2AAgent has no public card fetch; call it once connect has one.
        await agent._a2a_client()
        _say(texts.card_line)
        if args.ping:
            turn = await agent.send(settings.defaults.ping_message)
            _say(texts.ping_line.format(text=turn.text))


def _is_loopback(host: str) -> bool:
    """Return whether host is localhost or a loopback IP address."""
    if host == _LOCALHOST:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def run_serve(args: argparse.Namespace, builders: Builders) -> None:
    """Serve the agent's [bridge] over HTTP until interrupted."""
    config = load_agent(args.agent)
    url = settings.defaults.serve_url.format(host=args.host, port=args.port)
    app = app_from_config(config, url=url)
    if not _is_loopback(args.host):
        _warn(settings.texts.open_bind.format(host=args.host))
    builders.serve(app, args.host, args.port)


def _examples(source: CaseSource, path: Path) -> list[DatasetExample]:
    """Return every case of source, validated; PreflightError if there is none."""
    examples = list(source.examples())
    if not examples:
        raise PreflightError(settings.texts.no_rows.format(path=path))
    return examples


def _golden_graders(
    args: argparse.Namespace,
    builders: Builders,
    config: AgentConfig,
    examples: Sequence[DatasetExample],
) -> Graders:
    """Return the agent's golden graders, the criteria judge only if a case needs it."""
    criteria = any(example["outputs"].get("criteria") for example in examples)
    return build_graders(
        config,
        model_factory=builders.model_factory,
        criteria=criteria,
        criteria_model=args.criteria_model,
    )


def _attack_evaluators() -> tuple[Callable[..., Any], ...]:
    """Return the red-team evaluators: attack_succeeded and turns_to_break."""
    return (
        create_async_attack_succeeded_evaluator(),
        create_async_turns_to_break_evaluator(),
    )


def _dataset_name(config: AgentConfig, source: CaseSource) -> str:
    """Return <[langsmith] dataset_prefix, else the agent's name>/<source name>."""
    prefix = config.section("langsmith").get(_DATASET_PREFIX_KEY) or config.name
    return settings.defaults.dataset_name.format(prefix=prefix, source=source.name)


def _experiment_config(
    args: argparse.Namespace, config: AgentConfig
) -> ExperimentConfig:
    """Return the experiment settings the run options ask for."""
    defaults = settings.defaults
    prefix = args.prefix or defaults.experiment_prefix.format(
        agent=config.name, command=args.command
    )
    return ExperimentConfig(
        experiment_prefix=prefix,
        max_concurrency=args.concurrency,
        num_repetitions=args.repetitions,
        upload_results=not args.local,
        limit=args.limit,
    )


def _metadata(args: argparse.Namespace, config: AgentConfig) -> dict[str, Any]:
    """Return the experiment metadata: agent, command, data file and history mode."""
    keys = settings.metadata_keys
    return {
        keys.agent: config.name,
        keys.command: args.command,
        keys.data: None if args.data is None else args.data.name,
        # replay only approximates history, so a run records which mode it used.
        keys.history: config.agent.history,
    }


async def _run_experiment(
    args: argparse.Namespace,
    builders: Builders,
    *,
    config: AgentConfig,
    source: CaseSource,
    target: Target,
    graders: Graders,
) -> dict[str, float | None]:
    """Run one experiment, write results/<run_id>/, print it and return its summary.

    --local runs on the offline client, which sends nothing; otherwise the cases
    are mirrored into a LangSmith dataset and the experiment is uploaded.
    """
    client = builders.local_client() if args.local else builders.client()
    sink = LangSmithDatasetSync(client, namespace=settings.cli.dataset_namespace)
    metadata = _metadata(args, config)
    report = await ExperimentRunner(client=client, sink=sink).run(
        source=source,
        dataset_name=_dataset_name(config, source),
        target=target,
        evaluators=graders.evaluators,
        summary_evaluators=graders.summary_evaluators,
        config=_experiment_config(args, config),
        metadata=metadata,
    )
    pass_keys, segment_by = graders.pass_keys, args.segment_by
    rates = summarize(report.rows, pass_keys=pass_keys, segment_by=segment_by)
    summary = {**rates, **report.summary}
    keys = settings.metadata_keys
    meta = {
        **metadata,
        keys.experiment: report.experiment_name,
        keys.dataset: report.dataset_name,
        keys.url: report.url,
    }
    results_dir = args.results_dir
    run_dir = write_run(report, results_dir=results_dir, summary=summary, meta=meta)
    _say(replace(report, summary=summary).render())
    _say(settings.texts.results_line.format(path=run_dir))
    return summary


async def run_eval(args: argparse.Namespace, builders: Builders) -> None:
    """Score the agent on the golden cases of --data, then write and print the run."""
    config = load_agent(args.agent)
    source = JsonlGoldenSource(args.data)
    graders = _golden_graders(args, builders, config, _examples(source, args.data))
    async with builders.agent(config) as agent:
        await _run_experiment(
            args,
            builders,
            config=config,
            source=source,
            target=GoldenTarget(agent),
            graders=graders,
        )


async def run_simulate(args: argparse.Namespace, builders: Builders) -> None:
    """Run each scenario of --scenarios as a conversation, then write and print it.

    The golden graders grade each whole conversation, beside the red-team ones.
    """
    config = load_agent(args.agent)
    source = ScenarioSource(args.data)
    graders = _golden_graders(args, builders, config, _examples(source, args.data))
    evaluators = (*graders.evaluators, *_attack_evaluators())
    async with builders.agent(config) as agent:
        target = SimulationTarget(
            agent,
            config=config,
            model_factory=builders.model_factory,
            max_turns=args.max_turns,
        )
        await _run_experiment(
            args,
            builders,
            config=config,
            source=source,
            target=target,
            graders=replace(graders, evaluators=evaluators),
        )


def _redteam_source(args: argparse.Namespace, config: AgentConfig) -> CaseSource:
    """Return the attack cases: --personas, else [redteam], each attempts times."""
    redteam = agent_table(config, "redteam", RedteamSection)
    personas = load_personas(config)
    try:
        return redteam_cases(
            personas,
            names=args.personas or redteam.personas,
            attempts=args.attempts or redteam.attempts,
        )
    except ValueError as error:
        # A persona that is unknown or benign is bad input, not a crash.
        raise PreflightError(str(error)) from None


def _warn_if_unjudged(summary: dict[str, float | None]) -> None:
    """Warn loudly when some attacks were never judged, so ASR is uncertain."""
    key = simulation_settings.keys.attack_succeeded
    # A key no row scored is missing from the summary, which is coverage 0.
    coverage = summary.get(f"{key}_coverage") or 0.0
    if coverage < 1:
        _warn(settings.texts.low_coverage.format(key=key, coverage=coverage))


async def run_redteam(args: argparse.Namespace, builders: Builders) -> None:
    """Attack the agent with red-team personas, then write and print the run.

    attack_succeeded is the pass key, so pass_rate and pass_rate[persona=...]
    are attack success rates; an errored conversation counts as a failed attack.
    """
    config = load_agent(args.agent)
    source = _redteam_source(args, config)
    attack_key = simulation_settings.keys.attack_succeeded
    graders = Graders(evaluators=_attack_evaluators(), pass_keys=(attack_key,))
    async with builders.agent(config) as agent:
        target = SimulationTarget(
            agent,
            config=config,
            model_factory=builders.model_factory,
            max_turns=args.max_turns,
        )
        summary = await _run_experiment(
            args,
            builders,
            config=config,
            source=source,
            target=target,
            graders=graders,
        )
    _warn_if_unjudged(summary)


def _positive_int(text: str) -> int:
    """Return text as a whole number of at least one, for a count option."""
    if not text.isdecimal() or int(text) < 1:
        message = settings.texts.not_positive_int.format(text=text)
        raise argparse.ArgumentTypeError(message)
    return int(text)


def _names(text: str) -> list[str]:
    """Return the separated names in text, each stripped, blanks dropped."""
    names = text.split(settings.defaults.persona_separator)
    return [name.strip() for name in names if name.strip()]


def _add_agent(parser: argparse.ArgumentParser) -> None:
    """Add the agent argument every command takes."""
    help_texts = settings.help
    parser.add_argument(
        "agent", metavar=help_texts.metavars.agent, help=help_texts.agent
    )


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    """Add the options every experiment command passes to ExperimentConfig."""
    help_texts, defaults = settings.help, settings.defaults
    metavars = help_texts.metavars
    parser.add_argument("--local", action="store_true", help=help_texts.local)
    parser.add_argument(
        "--limit", type=_positive_int, metavar=metavars.count, help=help_texts.limit
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_int,
        default=core_settings.langsmith.repetitions,
        metavar=metavars.count,
        help=help_texts.repetitions,
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=defaults.max_concurrency,
        metavar=metavars.count,
        help=help_texts.concurrency,
    )
    parser.add_argument("--prefix", metavar=metavars.prefix, help=help_texts.prefix)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(defaults.results_dir),
        metavar=metavars.directory,
        help=help_texts.results_dir,
    )


def _add_criteria_model(parser: argparse.ArgumentParser) -> None:
    """Add the option naming the expected.criteria judge's model."""
    help_texts = settings.help
    parser.add_argument(
        "--criteria-model",
        metavar=help_texts.metavars.model,
        help=help_texts.criteria_model,
    )


def _add_max_turns(parser: argparse.ArgumentParser) -> None:
    """Add the option capping each simulated conversation's user messages."""
    help_texts = settings.help
    parser.add_argument(
        "--max-turns",
        type=_positive_int,
        metavar=help_texts.metavars.count,
        help=help_texts.max_turns,
    )


def _add_check(parser: argparse.ArgumentParser) -> None:
    """Add the check command's arguments."""
    _add_agent(parser)
    parser.add_argument("--ping", action="store_true", help=settings.help.ping)
    parser.set_defaults(run=run_check, errors=_CHECK_ERRORS, data=None)


def _add_serve(parser: argparse.ArgumentParser) -> None:
    """Add the serve command's arguments."""
    help_texts, defaults = settings.help, settings.defaults
    metavars = help_texts.metavars
    _add_agent(parser)
    parser.add_argument(
        "--host", default=defaults.host, metavar=metavars.host, help=help_texts.host
    )
    parser.add_argument(
        "--port",
        type=_positive_int,
        default=defaults.port,
        metavar=metavars.port,
        help=help_texts.port,
    )
    parser.set_defaults(run=run_serve, errors=_INPUT_ERRORS, data=None)


def _add_eval(parser: argparse.ArgumentParser) -> None:
    """Add the eval command's arguments."""
    help_texts = settings.help
    _add_agent(parser)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        metavar=help_texts.metavars.path,
        help=help_texts.data,
    )
    _add_criteria_model(parser)
    _add_run_options(parser)
    parser.set_defaults(
        run=run_eval, errors=_INPUT_ERRORS, segment_by=settings.segments.eval
    )


def _add_simulate(parser: argparse.ArgumentParser) -> None:
    """Add the simulate command's arguments."""
    help_texts = settings.help
    _add_agent(parser)
    parser.add_argument(
        "--scenarios",
        dest="data",
        type=Path,
        required=True,
        metavar=help_texts.metavars.path,
        help=help_texts.scenarios,
    )
    _add_max_turns(parser)
    _add_criteria_model(parser)
    _add_run_options(parser)
    parser.set_defaults(
        run=run_simulate, errors=_INPUT_ERRORS, segment_by=settings.segments.simulate
    )


def _add_redteam(parser: argparse.ArgumentParser) -> None:
    """Add the redteam command's arguments."""
    help_texts = settings.help
    metavars = help_texts.metavars
    _add_agent(parser)
    parser.add_argument(
        "--personas", type=_names, metavar=metavars.names, help=help_texts.personas
    )
    parser.add_argument(
        "--attempts",
        type=_positive_int,
        metavar=metavars.count,
        help=help_texts.attempts,
    )
    _add_max_turns(parser)
    _add_run_options(parser)
    parser.set_defaults(
        run=run_redteam,
        errors=_INPUT_ERRORS,
        segment_by=settings.segments.redteam,
        data=None,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of every command; each sets run, errors and data."""
    help_texts = settings.help
    parser = argparse.ArgumentParser(
        prog=settings.cli.prog, description=help_texts.description
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar=help_texts.metavars.command
    )
    _add_check(commands.add_parser("check", help=help_texts.check))
    _add_serve(commands.add_parser("serve", help=help_texts.serve))
    _add_eval(commands.add_parser("eval", help=help_texts.eval))
    _add_simulate(commands.add_parser("simulate", help=help_texts.simulate))
    _add_redteam(commands.add_parser("redteam", help=help_texts.redteam))
    return parser


def main(
    argv: Sequence[str] | None = None, *, builders: Builders = DEFAULT_BUILDERS
) -> int:
    """Parse argv, check prerequisites, run the command and return its exit code.

    A missing LangSmith key without --local, or a missing data file, returns
    input_error_exit_code with one line on stderr before anything is built. A
    bad agent folder or data row, an empty data file, an unknown persona and an
    AgentError, such as an unset token variable, end the same way; check also
    reports module paths that do not resolve, bad [bridge] options and an agent
    card it cannot fetch. Any other error propagates with its traceback.
    """
    args = build_parser().parse_args(argv)
    try:
        _preflight(args)
        outcome = args.run(args, builders)
        # serve is synchronous: uvicorn.run owns its event loop and its Ctrl-C.
        if asyncio.iscoroutine(outcome):
            asyncio.run(outcome)
    except args.errors as error:
        _warn(settings.texts.error_line.format(prog=settings.cli.prog, error=error))
        return settings.cli.input_error_exit_code
    return 0


__all__ = [
    "DEFAULT_BUILDERS",
    "Builders",
    "PreflightError",
    "build_parser",
    "main",
    "run_check",
    "run_eval",
    "run_redteam",
    "run_serve",
    "run_simulate",
]
