"""Command line and composition root of the jailbreak detection and simulation evals."""

import argparse
import asyncio
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langsmith import Client
from src.toolsets.jailbreak_detection.settings import settings as detector_settings

from trajectory_evals.suites.jailbreak.data import (
    CaseFormatError,
    JsonlCaseSource,
    LangSmithDatasetSync,
)
from trajectory_evals.suites.jailbreak.detection import (
    DetectionTarget,
    create_async_detection_match_evaluator,
    create_async_escalation_evaluator,
    create_async_fail_open_evaluator,
    create_detection_summary_evaluators,
)
from trajectory_evals.suites.jailbreak.protocols import CaseSource, Detector
from trajectory_evals.suites.jailbreak.runner import (
    ExperimentConfig,
    ExperimentMetadata,
    ExperimentRunner,
)
from trajectory_evals.suites.jailbreak.settings import settings
from trajectory_evals.suites.jailbreak.simulation import (
    PERSONAS,
    SimulationCaseSource,
    SimulationTarget,
    create_async_attack_attempted_evaluator,
    create_async_persona_faithful_evaluator,
    create_async_simulation_correct_evaluator,
    create_async_turns_to_block_evaluator,
    create_simulation_summary_evaluators,
)


class PreflightError(Exception):
    """A missing credential or data file that would fail a run or falsify its scores."""


@dataclass(frozen=True)
class Preflight:
    """Checks credentials and data files before any model or LangSmith call."""

    environ: Mapping[str, str]
    has_aws_credentials: Callable[[], bool]

    def check(self, *, data: Sequence[Path], required_env: Mapping[str, str]) -> None:
        """Raise PreflightError naming the first missing credential or data file."""
        texts = settings.texts
        if not self.has_aws_credentials():
            raise PreflightError(texts.no_aws_credentials)
        for name, message in required_env.items():
            if not self.environ.get(name):
                raise PreflightError(message.format(env=name))
        if not data:
            data_dir = settings.paths.data_path
            raise PreflightError(texts.no_data.format(data_dir=data_dir))
        missing = [str(path) for path in data if not path.is_file()]
        if missing:
            paths = texts.list_separator.join(missing)
            raise PreflightError(texts.data_not_found.format(paths=paths))


@dataclass(frozen=True)
class Builders:
    """Factories for the uploading and offline clients, detector and chat models."""

    client: Callable[[], Client]
    local_client: Callable[[], Client]
    detector: Callable[[], Detector]
    chat_model: Callable[[str], BaseChatModel]


def _has_aws_credentials() -> bool:
    """Return whether boto3 resolves AWS credentials, refreshing an SSO login."""
    try:
        credentials = boto3.Session().get_credentials()
        if credentials is not None:
            credentials.get_frozen_credentials()
    except (BotoCoreError, ClientError):
        return False
    return credentials is not None


def _build_local_client() -> Client:
    """Return a client with no tracing thread, so a local run sends nothing."""
    return Client(auto_batch_tracing=False, tracing_sampling_rate=0.0)


def _build_detector() -> Detector:
    """Return the real detection cascade on a Bedrock client in the AWS region."""
    # Imported on use, so cli imports without aioboto3, which the boto3 pin excludes.
    from src.toolsets.jailbreak_detection.bedrock import BedrockClient
    from src.toolsets.jailbreak_detection.factory import (
        build_jailbreak_detection_service,
    )

    region = os.environ.get(
        settings.cli.aws_region_env, settings.cli.default_aws_region
    )
    # src is Any to mypy without the detector repo on MYPYPATH.
    return build_jailbreak_detection_service(BedrockClient(detector_settings, region))  # type: ignore[no-any-return]


def _build_chat_model(model: str) -> BaseChatModel:
    """Return the OpenAI chat model with the given name."""
    return ChatOpenAI(model=model)


DEFAULT_PREFLIGHT = Preflight(
    environ=os.environ, has_aws_credentials=_has_aws_credentials
)
DEFAULT_BUILDERS = Builders(
    client=Client,
    local_client=_build_local_client,
    detector=_build_detector,
    chat_model=_build_chat_model,
)


def _build_runner(args: argparse.Namespace, builders: Builders) -> ExperimentRunner:
    """Return a runner on the offline client if local, else the uploading one."""
    client = builders.local_client() if args.local else builders.client()
    return ExperimentRunner(client=client, sink=LangSmithDatasetSync(client))


def _required_env(args: argparse.Namespace) -> dict[str, str]:
    """Return each variable the command needs, mapped to the error when it is unset."""
    cli, texts = settings.cli, settings.texts
    required = {} if args.local else {cli.langsmith_api_key_env: texts.no_langsmith_key}
    if args.needs_openai:
        required[cli.openai_api_key_env] = texts.no_openai_key
    return required


def _dataset_name(prefix: str, source: CaseSource) -> str:
    """Return the LangSmith dataset name for a source under the eval's prefix."""
    return f"{prefix}/{source.name}"


def _experiment_config(
    args: argparse.Namespace, limit: int | None = None
) -> ExperimentConfig:
    """Return the experiment settings the run options ask for."""
    return ExperimentConfig(
        experiment_prefix=args.experiment_prefix,
        max_concurrency=args.max_concurrency,
        num_repetitions=args.repetitions,
        upload_results=not args.local,
        limit=limit,
    )


def _metadata(
    path: Path, source: CaseSource, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return a data file's experiment metadata; refuse a file with no rows."""
    row_count = sum(1 for _ in source.examples())
    if not row_count:
        raise PreflightError(settings.texts.no_rows.format(path=path))
    return ExperimentMetadata.collect(
        dataset_file=path, row_count=row_count, extra=extra
    ).as_dict()


async def run_detection(args: argparse.Namespace, builders: Builders) -> None:
    """Validate every data file, then score each as its own experiment and print it."""
    sources = [JsonlCaseSource(path) for path in args.data]
    metadata = [
        _metadata(path, source) for path, source in zip(args.data, sources, strict=True)
    ]
    runner = _build_runner(args, builders)
    target = DetectionTarget(builders.detector())
    for source, source_metadata in zip(sources, metadata, strict=True):
        report = await runner.run(
            source=source,
            dataset_name=_dataset_name(settings.detection.dataset_prefix, source),
            target=target,
            evaluators=[
                create_async_detection_match_evaluator(),
                create_async_escalation_evaluator(),
                create_async_fail_open_evaluator(),
            ],
            summary_evaluators=create_detection_summary_evaluators(),
            config=_experiment_config(args, args.limit),
            metadata=source_metadata,
        )
        print(report.render())


async def run_simulation(args: argparse.Namespace, builders: Builders) -> None:
    """Simulate each persona on sampled openers as one experiment and print it."""
    path = args.data[0]
    source = SimulationCaseSource(
        JsonlCaseSource(path), args.personas, args.sample, args.seed
    )
    metadata = _metadata(path, source, extra=_simulation_extra(args))
    target = SimulationTarget(
        detector=builders.detector(),
        user_client=builders.chat_model(args.user_model),
        responder=builders.chat_model(args.responder_model),
        judge=builders.chat_model(args.judge_model),
        max_turns=args.max_turns,
    )
    report = await _build_runner(args, builders).run(
        source=source,
        dataset_name=_dataset_name(settings.simulation.dataset_prefix, source),
        target=target,
        evaluators=[
            create_async_simulation_correct_evaluator(),
            create_async_turns_to_block_evaluator(),
            create_async_attack_attempted_evaluator(),
            create_async_persona_faithful_evaluator(),
        ],
        summary_evaluators=create_simulation_summary_evaluators(),
        config=_experiment_config(args),
        metadata=metadata,
    )
    print(report.render())


def _simulation_extra(args: argparse.Namespace) -> dict[str, Any]:
    """Return the simulation settings recorded beside the detector's own metadata."""
    keys = settings.cli.metadata_keys
    return {
        keys.personas: args.personas,
        keys.sample: args.sample,
        keys.seed: args.seed,
        keys.max_turns: args.max_turns,
        keys.user_model: args.user_model,
        keys.responder_model: args.responder_model,
        keys.judge_model: args.judge_model,
    }


def _positive_int(text: str) -> int:
    """Return text as a whole number of at least one, for a count option."""
    if not text.isdecimal() or int(text) < 1:
        message = settings.texts.not_positive_int.format(text=text)
        raise argparse.ArgumentTypeError(message)
    return int(text)


class _UniqueNames(argparse.Action):
    """Stores an option's values once each, in the order they were first given."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Set the option to its values with every repeat dropped."""
        setattr(namespace, self.dest, list(dict.fromkeys(values or ())))


def _add_run_options(
    parser: argparse.ArgumentParser, *, max_concurrency: int, experiment_prefix: str
) -> None:
    """Add the options every command passes to ExperimentConfig."""
    help_texts = settings.help
    metavars = help_texts.metavars
    parser.add_argument(
        "--max-concurrency",
        type=_positive_int,
        default=max_concurrency,
        metavar=metavars.count,
        help=help_texts.max_concurrency,
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_int,
        default=settings.langsmith.repetitions,
        metavar=metavars.count,
        help=help_texts.repetitions,
    )
    parser.add_argument("--local", action="store_true", help=help_texts.local)
    parser.add_argument(
        "--experiment-prefix",
        default=experiment_prefix,
        metavar=metavars.prefix,
        help=help_texts.experiment_prefix,
    )


def _add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the detection command's options."""
    paths, detection, help_texts = settings.paths, settings.detection, settings.help
    metavars = help_texts.metavars
    parser.add_argument(
        "--data",
        nargs="+",
        type=Path,
        default=sorted(paths.data_path.glob(paths.data_glob)),
        metavar=metavars.path,
        help=help_texts.detection_data.format(data_dir=paths.data_dir),
    )
    parser.add_argument(
        "--limit", type=_positive_int, metavar=metavars.count, help=help_texts.limit
    )
    _add_run_options(
        parser,
        max_concurrency=detection.max_concurrency,
        experiment_prefix=detection.experiment_prefix,
    )
    parser.set_defaults(run=run_detection, needs_openai=False)


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the simulation command's options."""
    paths, simulation, help_texts = settings.paths, settings.simulation, settings.help
    metavars = help_texts.metavars
    data_help = help_texts.simulation_data
    parser.add_argument(
        "--data",
        nargs=1,
        type=Path,
        default=[paths.data_path / paths.simulation_data],
        metavar=metavars.path,
        help=data_help.format(data_dir=paths.data_dir, file=paths.simulation_data),
    )
    parser.add_argument(
        "--sample",
        type=_positive_int,
        default=simulation.sample,
        metavar=metavars.count,
        help=help_texts.sample,
    )
    parser.add_argument(
        "--seed", type=int, default=simulation.seed, help=help_texts.seed
    )
    names = help_texts.persona_separator.join(PERSONAS)
    parser.add_argument(
        "--personas",
        nargs="+",
        action=_UniqueNames,
        choices=list(PERSONAS),
        default=list(PERSONAS),
        metavar=metavars.persona,
        help=help_texts.personas.format(names=names),
    )
    parser.add_argument(
        "--max-turns",
        type=_positive_int,
        default=simulation.max_turns,
        metavar=metavars.count,
        help=help_texts.max_turns,
    )
    _add_model_options(parser)
    _add_run_options(
        parser,
        max_concurrency=simulation.max_concurrency,
        experiment_prefix=simulation.experiment_prefix,
    )
    parser.set_defaults(run=run_simulation, needs_openai=True)


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    """Add the OpenAI model names of the simulated user, responder and judge."""
    models, help_texts = settings.models, settings.help
    for option, default, help_text in (
        ("--user-model", models.user, help_texts.user_model),
        ("--responder-model", models.responder, help_texts.responder_model),
        ("--judge-model", models.judge, help_texts.judge_model),
    ):
        parser.add_argument(
            option,
            default=default,
            metavar=help_texts.metavars.model,
            help=help_text,
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the detection and simulation commands."""
    help_texts = settings.help
    parser = argparse.ArgumentParser(
        prog=settings.cli.prog, description=help_texts.description
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar=help_texts.metavars.command
    )
    _add_detection_arguments(
        commands.add_parser("detection", help=help_texts.detection)
    )
    _add_simulation_arguments(
        commands.add_parser("simulation", help=help_texts.simulation)
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    preflight: Preflight = DEFAULT_PREFLIGHT,
    builders: Builders = DEFAULT_BUILDERS,
) -> int:
    """Parse argv, check prerequisites, run the chosen command and return its code."""
    args = build_parser().parse_args(argv)
    try:
        preflight.check(data=args.data, required_env=_required_env(args))
        asyncio.run(args.run(args, builders))
    except (PreflightError, CaseFormatError) as error:
        line = settings.texts.error_line.format(prog=settings.cli.prog, error=error)
        print(line, file=sys.stderr)
        return settings.cli.input_error_exit_code
    return 0


__all__ = [
    "DEFAULT_BUILDERS",
    "DEFAULT_PREFLIGHT",
    "Builders",
    "Preflight",
    "PreflightError",
    "build_parser",
    "main",
    "run_detection",
    "run_simulation",
]
