"""Command line and composition root of every trajectory_evals command."""

import argparse
import asyncio
import ipaddress
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langsmith import Client, tracing_context
from langsmith.utils import LangSmithNotFoundError
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
from trajectory_evals.core.jsonl import CaseFormatError, describe_error
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
from trajectory_evals.langsmith_io import (
    HARVEST_DIR,
    LangSmithClient,
    LangSmithSection,
    harvest_runs,
    langsmith_section,
    online_evaluators,
    pull_dataset,
    push_dataset,
    score_runs,
)
from trajectory_evals.optimize import (
    REPORT_FILE,
    attack_source,
    optimize,
    optimize_table,
    sweep,
    variant_tables,
)
from trajectory_evals.simulation import (
    Persona,
    RedteamSection,
    ScenarioSource,
    SimulationTarget,
    agent_table,
    create_async_attack_succeeded_evaluator,
    create_async_defended_evaluator,
    create_async_turns_to_break_evaluator,
    load_personas,
    redteam_cases,
)
from trajectory_evals.simulation.settings import settings as simulation_settings

LANGSMITH_API_KEY_ENV = "LANGSMITH_API_KEY"
INPUT_ERROR_EXIT_CODE = 2
_LOCALHOST = "localhost"
_IPV6 = 6
_ROWS_KEY = "rows"  # summarize's row count, which the report prints as runs
_DISPATCH_KEYS = frozenset({"run", "errors"})  # parser fields that are not options
_MODEL_NAME = TypeAdapter(ModelName)


class PreflightError(Exception):
    """Bad input found before a run: a missing key, file or case, or a bad name."""


# What every command reports as bad input: a missing prerequisite, a bad agent
# folder, data row or option value, or an agent that fails to start, such as
# one whose token variable is unset.
_INPUT_ERRORS = (
    PreflightError,
    AgentConfigError,
    CaseFormatError,
    AgentError,
    ValidationError,
)
# optimize and sweep also raise ValueError for bad splits, before any agent call.
_OPTIMIZE_ERRORS = (*_INPUT_ERRORS, ValueError)
# The langsmith commands refuse bad options, an existing or tracked output file
# and an unknown dataset or project before writing anything.
_LANGSMITH_ERRORS = (
    *_INPUT_ERRORS,
    ValueError,
    FileExistsError,
    LangSmithNotFoundError,
)
# check also reports a module:attr that does not resolve and bad table values,
# since finding those is its job; a card it cannot fetch is an AgentError.
_CHECK_ERRORS = (*_INPUT_ERRORS, ValueError, ImportError, AttributeError)


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


def _error_text(error: BaseException) -> str:
    """Return the error's message; a ValidationError's never repeats the values."""
    if isinstance(error, ValidationError):
        return describe_error(error)
    return str(error)


def _score(value: float | None) -> str:
    """Return a score as the report prints it, or the undefined label if None."""
    report = core_settings.langsmith
    if value is None:
        return report.undefined_score
    return f"{value:.{report.score_decimals}f}"


def _preflight(args: argparse.Namespace) -> None:
    """Raise PreflightError for a missing LangSmith key or data file."""
    texts = settings.texts
    # check, serve and --local runs never reach LangSmith; langsmith's always do.
    online = not getattr(args, "local", True)
    if online and not os.environ.get(LANGSMITH_API_KEY_ENV):
        raise PreflightError(texts.no_langsmith_key.format(env=LANGSMITH_API_KEY_ENV))
    if args.data is not None and not args.data.is_file():
        raise PreflightError(texts.data_not_found.format(path=args.data))


def _start_failed(template: str, name: str, error: Exception) -> PreflightError:
    """Return the error for a model or agent that could not be built."""
    return PreflightError(template.format(name=name, error=_error_text(error)))


def _guarded_models(model_factory: ModelFactory) -> ModelFactory:
    """Return model_factory, raising PreflightError for a model it cannot build."""

    def build(name: str) -> BaseChatModel:
        """Return the named model, or raise PreflightError saying why not."""
        try:
            return model_factory(name)
        # Providers raise their own types for a missing key or package.
        except Exception as error:
            template = settings.texts.model_failed
            raise _start_failed(template, name, error) from error

    return build


def _guarded_agent(
    agent: Callable[[AgentConfig], A2AAgent],
) -> Callable[[AgentConfig], A2AAgent]:
    """Return agent, raising PreflightError for an agent it cannot build."""

    def build(config: AgentConfig) -> A2AAgent:
        """Return the agent, or raise PreflightError saying why not."""
        try:
            return agent(config)
        # A [bridge] responder factory raises whatever its own model or options do.
        except Exception as error:
            template = settings.texts.agent_failed
            raise _start_failed(template, config.name, error) from error

    return build


def _model_stand_in(name: str) -> BaseChatModel:
    """Return a model check never calls, once name is a provider:model string."""
    try:
        _MODEL_NAME.validate_python(name)
    except ValidationError:
        raise AgentConfigError(settings.texts.bad_model.format(name=name)) from None
    # check validates config only, so no API key or provider package is needed.
    return GenericFakeChatModel(messages=iter(()))


def _require_personas(
    examples: Sequence[DatasetExample], personas: Mapping[str, Persona], path: Path
) -> None:
    """Raise PreflightError if a scenario names a persona the agent does not have."""
    named = {example["inputs"]["persona"] for example in examples}
    unknown = sorted(named - personas.keys())
    if unknown:
        message = settings.texts.unknown_personas
        known = sorted(personas)
        raise PreflightError(message.format(path=path, names=unknown, known=known))


def _dataset_counts(
    config: AgentConfig, personas: Mapping[str, Persona]
) -> list[tuple[Path, int]]:
    """Return each datasets file of the folder, relative, with its case count.

    A file whose name holds the scenario marker is read as scenarios, whose
    personas must exist; any other as golden cases.
    """
    check = settings.check
    counts = []
    for path in sorted((config.root / check.datasets_dir).glob(check.datasets_glob)):
        if check.scenario_marker in path.name:
            examples = list(ScenarioSource(path).examples())
            _require_personas(examples, personas, path)
        else:
            examples = list(JsonlGoldenSource(path).examples())
        counts.append((path.relative_to(config.root), len(examples)))
    return counts


def _check_run_tables(config: AgentConfig) -> None:
    """Validate the tables redteam, optimize, sweep and langsmith reject at start."""
    redteam = agent_table(config, "redteam", RedteamSection)
    personas = load_personas(config)
    redteam_cases(personas, names=redteam.personas, attempts=redteam.attempts)
    langsmith_section(config)
    if not (config.section("optimize") or config.section("variants")):
        return
    section = optimize_table(config)
    variant_tables(config, section)
    attack_source(config, section)


async def run_check(args: argparse.Namespace, builders: Builders) -> None:
    """Validate the agent's tables, datasets and card; with --ping, send a message.

    Every table is validated as the run commands read it, with model names
    checked as provider:model strings and no model built.
    """
    texts = settings.texts
    config = load_agent(args.agent)
    _say(texts.agent_line.format(name=config.name, root=config.root))
    build_graders(config, model_factory=_model_stand_in, criteria=False)
    for path, count in _dataset_counts(config, load_personas(config)):
        _say(texts.dataset_line.format(path=path, count=count))
    async with builders.agent(config) as agent:
        SimulationTarget(agent, config=config, model_factory=_model_stand_in)
        _check_run_tables(config)
        _say(texts.tables_line)
        await agent.card()
        _say(texts.card_line)
        if args.ping:
            # A ping is a probe, not an experiment, so it traces nowhere.
            with tracing_context(enabled=False):
                turn = await agent.send(settings.defaults.ping_message)
            _say(texts.ping_line.format(text=turn.text))


def _ip_address(
    host: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Return host as an IP address, or None for a host name."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def run_serve(args: argparse.Namespace, builders: Builders) -> None:
    """Serve the agent's [bridge] over HTTP until interrupted.

    The card advertises --url, else http://<host>:<port>/ with an IPv6 host in
    brackets; binding off loopback, or to every address without --url, warns.
    """
    texts = settings.texts
    config = load_agent(args.agent)
    address = _ip_address(args.host)
    host = f"[{args.host}]" if address and address.version == _IPV6 else args.host
    url = args.url or settings.defaults.serve_url.format(host=host, port=args.port)
    app = app_from_config(config, url=url)
    if not (args.host == _LOCALHOST or (address and address.is_loopback)):
        _warn(texts.open_bind.format(host=args.host))
    if args.url is None and address and address.is_unspecified:
        _warn(texts.needs_url.format(url=url))
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
    """Return the agent's golden graders; criteria only if a case run needs it."""
    run = examples[: args.limit]
    criteria = any(example["outputs"].get("criteria") for example in run)
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


def _dataset_name(config: AgentConfig, source_name: str) -> str:
    """Return <[langsmith] dataset_prefix, else the agent's name>/<source name>."""
    prefix = langsmith_section(config).dataset_prefix or config.name
    return settings.defaults.dataset_name.format(prefix=prefix, source=source_name)


@contextmanager
def _experiment_runner(
    args: argparse.Namespace, builders: Builders, config: AgentConfig
) -> Iterator[ExperimentRunner]:
    """Yield the runner: with --local, on the offline client and tracing nowhere.

    Without --local, the cases are mirrored into LangSmith datasets under the
    [langsmith] namespace, so a case gets the id langsmith push gives it.
    """
    client = builders.local_client() if args.local else builders.client()
    namespace = langsmith_section(config).namespace
    sink = LangSmithDatasetSync(client, namespace=namespace)
    # Models called outside aevaluate, such as the agent's and the reflection
    # model, would otherwise trace to the default client, which sends.
    tracing: AbstractContextManager[None] = (
        tracing_context(client=client) if args.local else nullcontext()
    )
    with tracing:
        yield ExperimentRunner(client=client, sink=sink)


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


def _options(args: argparse.Namespace) -> dict[str, Any]:
    """Return the parsed command line, without its dispatch fields."""
    return {
        key: value for key, value in vars(args).items() if key not in _DISPATCH_KEYS
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

    The summary is summarize's scores, then each [graders] summary score whose
    key summarize does not report. meta.json holds the experiment metadata, its
    name, dataset and URL, and the run's options.
    """
    metadata = _metadata(args, config)
    with _experiment_runner(args, builders, config) as runner:
        report = await runner.run(
            source=source,
            dataset_name=_dataset_name(config, source.name),
            target=target,
            evaluators=graders.evaluators,
            summary_evaluators=graders.summary_evaluators,
            config=_experiment_config(args, config),
            metadata=metadata,
        )
    pass_keys, segment_by = graders.pass_keys, args.segment_by
    rates = summarize(report.rows, pass_keys=pass_keys, segment_by=segment_by)
    # summarize's keys win, so a [graders] summary key of the same name, such as
    # the jailbreak detector's error_rate, never redefines pass_rate or error_rate.
    extra = {key: value for key, value in report.summary.items() if key not in rates}
    summary = {**rates, **extra}
    keys = settings.metadata_keys
    meta = {
        **metadata,
        keys.experiment: report.experiment_name,
        keys.dataset: report.dataset_name,
        keys.url: report.url,
        keys.options: _options(args),
    }
    results_dir = args.results_dir
    run_dir = write_run(report, results_dir=results_dir, summary=summary, meta=meta)
    printed = {key: value for key, value in summary.items() if key != _ROWS_KEY}
    _say(replace(report, summary=printed).render())
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
    examples = _examples(source, args.data)
    _require_personas(examples, load_personas(config), args.data)
    graders = _golden_graders(args, builders, config, examples)
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
    if redteam.policy is None:
        raise PreflightError(settings.texts.no_policy)
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


def _report_attacks(summary: Mapping[str, float | None]) -> None:
    """Print the defended rate and the attack success rate; warn on unjudged ones."""
    key = simulation_settings.keys.attack_succeeded
    # A key no row scored is missing from the summary, which is coverage 0.
    coverage = summary.get(f"{key}_coverage") or 0.0
    texts = settings.texts
    line = texts.asr_line.format(
        defended=_score(summary.get("pass_rate")),
        asr=_score(summary.get(key)),
        coverage=_score(coverage),
        error_rate=_score(summary.get("error_rate")),
    )
    _say(line)
    if coverage < 1:
        _warn(texts.low_coverage.format(key=key, coverage=coverage))


async def run_redteam(args: argparse.Namespace, builders: Builders) -> None:
    """Attack the agent with red-team personas, then write and print the run.

    defended is the pass key, so pass_rate is the share of attacks defended, an
    attack whose target raised counting as not defended. The attack success
    rate is attack_succeeded's mean over the judged attacks.
    """
    config = load_agent(args.agent)
    source = _redteam_source(args, config)
    graders = Graders(
        evaluators=(*_attack_evaluators(), create_async_defended_evaluator()),
        pass_keys=(simulation_settings.keys.defended,),
    )
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
    _report_attacks(summary)


def _read_report(run_dir: Path) -> dict[str, Any]:
    """Return an optimize or sweep run's report.json."""
    report: dict[str, Any] = json.loads(
        (run_dir / REPORT_FILE).read_text(encoding="utf-8")
    )
    return report


async def run_optimize(args: argparse.Namespace, builders: Builders) -> None:
    """Hill-climb the agent's [optimize] prompts on --data, then print the verdict."""
    config = load_agent(args.agent)
    source = JsonlGoldenSource(args.data)
    _examples(source, args.data)
    async with builders.agent(config) as agent:
        with _experiment_runner(args, builders, config) as runner:
            run_dir = await optimize(
                config=config,
                source=source,
                agent=agent,
                model_factory=builders.model_factory,
                runner=runner,
                results_dir=args.results_dir,
                upload=args.upload_as_experiments,
            )
    report = _read_report(run_dir)
    line = settings.texts.optimize_line.format(
        verdict=report["verdict"],
        reason=report["reason"] or settings.texts.no_reason,
        best=report["best"],
        stop=report["stop_reason"],
    )
    _say(line)
    _say(settings.texts.results_line.format(path=run_dir / REPORT_FILE))


def _variant_line(record: Mapping[str, Any]) -> str:
    """Return one sweep candidate's line: cost, val pass_rate and verdict."""
    texts = settings.texts
    return texts.variant_line.format(
        id=record["id"],
        cost=record["cost"],
        pass_rate=_score(record["val"].get("pass_rate")),
        verdict=record["verdict"],
        reason=record["reason"] or texts.no_reason,
    )


async def run_sweep(args: argparse.Namespace, builders: Builders) -> None:
    """Sweep the baseline, each variant and --include on --data; print the table."""
    texts = settings.texts
    if args.include is not None and not args.include.is_dir():
        raise PreflightError(texts.folder_not_found.format(path=args.include))
    config = load_agent(args.agent)
    source = JsonlGoldenSource(args.data)
    _examples(source, args.data)
    async with builders.agent(config) as agent:
        with _experiment_runner(args, builders, config) as runner:
            run_dir = await sweep(
                config=config,
                source=source,
                agent=agent,
                model_factory=builders.model_factory,
                runner=runner,
                results_dir=args.results_dir,
                upload=args.upload_as_experiments,
                include=args.include,
            )
    report = _read_report(run_dir)
    for record in report["variants"]:
        _say(_variant_line(record))
    line = texts.sweep_line.format(
        verdict=report["verdict"],
        reason=report["reason"] or texts.no_reason,
        winner=report["winner"],
    )
    _say(line)
    _say(texts.results_line.format(path=run_dir / REPORT_FILE))


def _langsmith_client(builders: Builders) -> LangSmithClient:
    """Return the uploading client; never the offline one, which drops feedback."""
    return builders.client()


def _project(args: argparse.Namespace, section: LangSmithSection) -> str:
    """Return --project, else [langsmith] project; PreflightError if neither."""
    project = args.project or section.project
    if project is None:
        raise PreflightError(settings.texts.no_project)
    return project


def _since(args: argparse.Namespace, section: LangSmithSection) -> datetime:
    """Return the earliest start time of the runs to read."""
    days = args.since_days or section.since_days
    return datetime.now(UTC) - timedelta(days=days)


def run_pull(args: argparse.Namespace, builders: Builders) -> None:
    """Write the --dataset LangSmith dataset as golden JSONL at --out."""
    langsmith_section(load_agent(args.agent))
    path = pull_dataset(_langsmith_client(builders), args.dataset, out=args.out)
    _say(settings.texts.wrote_line.format(path=path))


def run_push(args: argparse.Namespace, builders: Builders) -> None:
    """Mirror the golden JSONL file --data into its LangSmith dataset."""
    config = load_agent(args.agent)
    section = langsmith_section(config)
    name = args.dataset or _dataset_name(config, JsonlGoldenSource(args.data).name)
    client = _langsmith_client(builders)
    pushed = push_dataset(client, args.data, name, namespace=section.namespace)
    _say(settings.texts.pushed_line.format(path=args.data, name=pushed))


async def run_harvest(args: argparse.Namespace, builders: Builders) -> None:
    """Write the project's recent root runs as golden-case candidates to review."""
    defaults = settings.defaults
    config = load_agent(args.agent)
    section = langsmith_section(config)
    stamp = datetime.now(UTC).strftime(defaults.stamp_format)
    default_out = HARVEST_DIR / defaults.harvest_file.format(
        agent=config.name, stamp=stamp
    )
    path = await harvest_runs(
        _langsmith_client(builders),
        _project(args, section),
        out=args.out or default_out,
        since=_since(args, section),
        limit=args.limit,
        filter=args.filter,
        only_errors=args.only_errors,
        feedback_key=args.feedback_key,
        feedback_below=args.feedback_below,
        inputs_key=section.inputs_key,
        outputs_key=section.outputs_key,
    )
    _say(settings.texts.wrote_line.format(path=path))


async def run_score(args: argparse.Namespace, builders: Builders) -> None:
    """Score the project's recent runs with the reference-free judges as feedback."""
    texts = settings.texts
    config = load_agent(args.agent)
    section = langsmith_section(config)
    evaluators = online_evaluators(config, model_factory=builders.model_factory)
    report = await score_runs(
        _langsmith_client(builders),
        _project(args, section),
        evaluators,
        since=_since(args, section),
        limit=args.limit,
        filter=args.filter,
        inputs_key=section.inputs_key,
        outputs_key=section.outputs_key,
        dry_run=args.dry_run,
    )
    _say(texts.runs_line.format(runs=report.runs))
    for key, counts in report.keys.items():
        _say(texts.key_line.format(key=key, **asdict(counts)))


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


def _add_data(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the required --data golden JSONL option."""
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        metavar=settings.help.metavars.path,
        help=help_text,
    )


def _add_results_dir(parser: argparse.ArgumentParser) -> None:
    """Add the option naming where a run's folder is written."""
    help_texts = settings.help
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(settings.defaults.results_dir),
        metavar=help_texts.metavars.directory,
        help=help_texts.results_dir,
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
    _add_results_dir(parser)


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
    parser.add_argument("--url", metavar=metavars.url, help=help_texts.url)
    parser.set_defaults(run=run_serve, errors=_INPUT_ERRORS, data=None)


def _add_eval(parser: argparse.ArgumentParser) -> None:
    """Add the eval command's arguments."""
    _add_agent(parser)
    _add_data(parser, settings.help.data)
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


def _add_optimize_options(parser: argparse.ArgumentParser) -> None:
    """Add the options optimize and sweep share."""
    help_texts = settings.help
    _add_agent(parser)
    _add_data(parser, help_texts.data)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--local", action="store_true", help=help_texts.local)
    modes.add_argument(
        "--upload-as-experiments",
        action="store_true",
        help=help_texts.upload_as_experiments,
    )
    _add_results_dir(parser)


def _add_optimize(parser: argparse.ArgumentParser) -> None:
    """Add the optimize command's arguments."""
    _add_optimize_options(parser)
    parser.set_defaults(run=run_optimize, errors=_OPTIMIZE_ERRORS)


def _add_sweep(parser: argparse.ArgumentParser) -> None:
    """Add the sweep command's arguments."""
    help_texts = settings.help
    _add_optimize_options(parser)
    parser.add_argument(
        "--include",
        type=Path,
        metavar=help_texts.metavars.directory,
        help=help_texts.include,
    )
    parser.set_defaults(run=run_sweep, errors=_OPTIMIZE_ERRORS)


def _add_run_query(parser: argparse.ArgumentParser) -> None:
    """Add the options harvest and score read runs with."""
    help_texts, metavars = settings.help, settings.help.metavars
    parser.add_argument("--project", metavar=metavars.project, help=help_texts.project)
    parser.add_argument(
        "--since-days",
        type=_positive_int,
        metavar=metavars.count,
        help=help_texts.since_days,
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=settings.defaults.run_limit,
        metavar=metavars.count,
        help=help_texts.run_limit,
    )
    parser.add_argument("--filter", metavar=metavars.filter, help=help_texts.filter)


def _langsmith_defaults(
    parser: argparse.ArgumentParser, run: Callable[..., Any]
) -> None:
    """Set a langsmith subcommand's run; it always needs LangSmith, never --local."""
    parser.set_defaults(run=run, errors=_LANGSMITH_ERRORS, local=False)


def _add_pull(parser: argparse.ArgumentParser) -> None:
    """Add the langsmith pull command's arguments."""
    help_texts, metavars = settings.help, settings.help.metavars
    _add_agent(parser)
    parser.add_argument(
        "--dataset", required=True, metavar=metavars.dataset, help=help_texts.dataset
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar=metavars.path,
        help=help_texts.pull_out,
    )
    _langsmith_defaults(parser, run_pull)
    parser.set_defaults(data=None)


def _add_push(parser: argparse.ArgumentParser) -> None:
    """Add the langsmith push command's arguments."""
    help_texts = settings.help
    _add_agent(parser)
    _add_data(parser, help_texts.data)
    parser.add_argument(
        "--dataset", metavar=help_texts.metavars.dataset, help=help_texts.push_dataset
    )
    _langsmith_defaults(parser, run_push)


def _add_harvest(parser: argparse.ArgumentParser) -> None:
    """Add the langsmith harvest command's arguments."""
    help_texts, metavars = settings.help, settings.help.metavars
    _add_agent(parser)
    _add_run_query(parser)
    parser.add_argument(
        "--only-errors", action="store_true", help=help_texts.only_errors
    )
    parser.add_argument(
        "--feedback-key", metavar=metavars.key, help=help_texts.feedback_key
    )
    parser.add_argument(
        "--feedback-below",
        type=float,
        metavar=metavars.score,
        help=help_texts.feedback_below,
    )
    parser.add_argument(
        "--out", type=Path, metavar=metavars.path, help=help_texts.harvest_out
    )
    _langsmith_defaults(parser, run_harvest)
    parser.set_defaults(data=None)


def _add_score(parser: argparse.ArgumentParser) -> None:
    """Add the langsmith score command's arguments."""
    _add_agent(parser)
    _add_run_query(parser)
    parser.add_argument("--dry-run", action="store_true", help=settings.help.dry_run)
    _langsmith_defaults(parser, run_score)
    parser.set_defaults(data=None)


def _add_langsmith(parser: argparse.ArgumentParser) -> None:
    """Add the langsmith command group: pull, push, harvest and score."""
    help_texts = settings.help
    commands = parser.add_subparsers(
        dest="langsmith_command", required=True, metavar=help_texts.metavars.command
    )
    _add_pull(commands.add_parser("pull", help=help_texts.pull))
    _add_push(commands.add_parser("push", help=help_texts.push))
    _add_harvest(commands.add_parser("harvest", help=help_texts.harvest))
    _add_score(commands.add_parser("score", help=help_texts.score))


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
    _add_optimize(commands.add_parser("optimize", help=help_texts.optimize))
    _add_sweep(commands.add_parser("sweep", help=help_texts.sweep))
    _add_langsmith(commands.add_parser("langsmith", help=help_texts.langsmith))
    return parser


def main(
    argv: Sequence[str] | None = None, *, builders: Builders = DEFAULT_BUILDERS
) -> int:
    """Parse argv, check prerequisites, run the command and return its exit code.

    A missing LangSmith key where one is needed, or a missing data file,
    returns INPUT_ERROR_EXIT_CODE with one line on stderr before anything is
    built. Each command's other bad input ends the same way: a bad agent folder,
    data row or option value, an empty data file, an unknown persona, a model
    that cannot be built and an agent that fails to start, plus what the command
    itself rejects up front. A ValidationError's line never repeats its values.
    Any other error propagates with its traceback.
    """
    args = build_parser().parse_args(argv)
    builders = replace(
        builders,
        agent=_guarded_agent(builders.agent),
        model_factory=_guarded_models(builders.model_factory),
    )
    try:
        _preflight(args)
        outcome = args.run(args, builders)
        # serve is synchronous: uvicorn.run owns its event loop and its Ctrl-C.
        if asyncio.iscoroutine(outcome):
            asyncio.run(outcome)
    except args.errors as error:
        # Some messages span lines, such as a provider list; the error is one line.
        text = " ".join(_error_text(error).split())
        _warn(settings.texts.error_line.format(prog=settings.cli.prog, error=text))
        return INPUT_ERROR_EXIT_CODE
    return 0


__all__ = [
    "DEFAULT_BUILDERS",
    "INPUT_ERROR_EXIT_CODE",
    "LANGSMITH_API_KEY_ENV",
    "Builders",
    "PreflightError",
    "build_parser",
    "main",
    "run_check",
    "run_eval",
    "run_harvest",
    "run_optimize",
    "run_pull",
    "run_push",
    "run_redteam",
    "run_score",
    "run_serve",
    "run_simulate",
    "run_sweep",
]
