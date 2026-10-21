"""Run one LangSmith experiment over a case source and report its scores."""

import asyncio
import hashlib
import itertools
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from langsmith import Client, aevaluate, schemas
from langsmith.evaluation._arunner import AsyncExperimentResults

from evals.jailbreak.protocols import CaseSource, DatasetSink
from evals.jailbreak.settings import settings as eval_settings
from src.toolsets.jailbreak_detection.settings import JailbreakDetectionSettings
from src.toolsets.jailbreak_detection.settings import settings as shipped_settings

Target = Callable[[Any], Awaitable[Mapping[str, Any]]]


def _hash_prompts(settings: JailbreakDetectionSettings) -> str:
    """Return a sha256 over both prompts, every tool text and every finding text."""
    texts = (
        settings.screen_prompt,
        settings.judge_prompt,
        settings.tools.model_dump_json(),
        settings.findings.model_dump_json(),
    )
    separator = eval_settings.langsmith.prompt_hash_separator
    return hashlib.sha256(separator.join(texts).encode()).hexdigest()


def _format_score(score: float | None) -> str:
    """Return score to the configured decimals, or the undefined label if None."""
    report = eval_settings.langsmith
    if score is None:
        return report.undefined_score
    return f"{score:.{report.score_decimals}f}"


def _wrap_target(
    target: Target,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Wrap target in the coroutine function returning a dict that aevaluate needs."""

    async def call_target(inputs: dict[str, Any]) -> dict[str, Any]:
        """Await target on one example's inputs and return its outputs as a dict."""
        return dict(await target(inputs))

    return call_target


def _build_local_examples(
    source: CaseSource, limit: int | None
) -> list[schemas.Example]:
    """Return the source's first limit cases, or all of them, as in-memory examples."""
    return [
        schemas.Example(
            id=uuid.uuid4(),
            inputs=case["inputs"],
            outputs=case["outputs"],
            metadata={
                **case["metadata"],
                eval_settings.data.case_id_key: case["case_id"],
            },
        )
        for case in itertools.islice(source.examples(), limit)
    ]


def _read_summary_scores(results: AsyncExperimentResults) -> dict[str, float | None]:
    """Return each summary score by key, read from aevaluate's private field."""
    summary = getattr(results, "_summary_results")
    return {
        result.key: None if result.score is None else float(result.score)
        for result in summary["results"]
    }


@dataclass(frozen=True)
class ExperimentConfig:
    """How one experiment runs: name prefix, concurrency, repeats and upload mode."""

    experiment_prefix: str
    max_concurrency: int
    num_repetitions: int = eval_settings.langsmith.repetitions
    upload_results: bool = True
    limit: int | None = None

    def __post_init__(self) -> None:
        """Reject a concurrency, repetition count or limit below one."""
        for name in ("max_concurrency", "num_repetitions", "limit"):
            value = getattr(self, name)
            if value is not None and value < 1:
                message = eval_settings.texts.at_least_one
                raise ValueError(message.format(name=name, value=value))


@dataclass(frozen=True)
class ExperimentMetadata:
    """The models, guardrail, prompts and dataset a jailbreak experiment ran against."""

    screen_model: str
    judge_model: str
    guardrail_id: str
    guardrail_version: str
    prompts_sha256: str
    dataset_file: str
    row_count: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def collect(
        cls,
        *,
        dataset_file: Path,
        row_count: int,
        extra: Mapping[str, Any] | None = None,
        settings: JailbreakDetectionSettings = shipped_settings,
    ) -> Self:
        """Gather one dataset file's metadata from the detection settings."""
        return cls(
            screen_model=settings.models.screen,
            judge_model=settings.models.judge,
            guardrail_id=settings.guardrail.id,
            guardrail_version=settings.guardrail.version,
            prompts_sha256=_hash_prompts(settings),
            dataset_file=dataset_file.name,
            row_count=row_count,
            extra=dict(extra or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the fields as flat experiment metadata, extra keys first."""
        recorded = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "extra"
        }
        return {**self.extra, **recorded}


@dataclass(frozen=True)
class ExperimentReport:
    """What one experiment produced: its name, dataset, URL, scores and run count."""

    experiment_name: str
    dataset_name: str
    url: str | None
    summary: Mapping[str, float | None]
    runs: int

    def render(self) -> str:
        """Return the report as the lines the CLI prints."""
        report = eval_settings.langsmith
        labels = report.report_labels
        fields: list[tuple[str, object]] = [
            (labels.experiment, self.experiment_name),
            (labels.dataset, self.dataset_name),
            (labels.url, self.url or report.not_uploaded),
            (labels.runs, self.runs),
        ]
        fields.extend(
            (key, _format_score(score)) for key, score in self.summary.items()
        )
        return "\n".join(
            report.report_line.format(label=label, value=value)
            for label, value in fields
        )


async def _name_experiment(
    results: AsyncExperimentResults, config: ExperimentConfig
) -> tuple[str, str | None]:
    """Return the experiment's name and comparison URL, or a local label and no URL."""
    if not config.upload_results:
        local_name = config.experiment_prefix + eval_settings.langsmith.local_suffix
        return local_name, None
    return results.experiment_name, await results.get_comparison_url()


class ExperimentRunner:
    """Mirrors a case source into a dataset and runs one aevaluate experiment on it."""

    def __init__(self, *, client: Client, sink: DatasetSink) -> None:
        """Keep the client every run uses, offline for a local run, and the sink."""
        self._client = client
        self._sink = sink

    async def run(
        self,
        *,
        source: CaseSource,
        dataset_name: str,
        target: Target,
        evaluators: Sequence[Callable[..., Any]],
        summary_evaluators: Sequence[Callable[..., Any]],
        config: ExperimentConfig,
        metadata: Mapping[str, Any],
    ) -> ExperimentReport:
        """Evaluate target on the source's cases and return the experiment's report."""
        if config.upload_results:
            dataset = await self._mirror_dataset(source, dataset_name, config.limit)
        else:
            dataset = _build_local_examples(source, config.limit)
        results = await aevaluate(
            _wrap_target(target),
            data=dataset,
            evaluators=evaluators,
            summary_evaluators=summary_evaluators,
            metadata=dict(metadata),
            experiment_prefix=config.experiment_prefix,
            max_concurrency=config.max_concurrency,
            num_repetitions=config.num_repetitions,
            client=self._client,
            upload_results=config.upload_results,
        )
        experiment_name, url = await _name_experiment(results, config)
        return ExperimentReport(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            url=url,
            summary=_read_summary_scores(results),
            runs=len(results),
        )

    async def _mirror_dataset(
        self, source: CaseSource, dataset_name: str, limit: int | None
    ) -> str | list[schemas.Example]:
        """Mirror the whole source, then return the dataset or its first limit cases."""
        synced_name = await asyncio.to_thread(
            self._sink.sync, dataset_name, source.examples()
        )
        if limit is None:
            return synced_name
        example_ids = [
            self._sink.example_id(synced_name, case["case_id"])
            for case in itertools.islice(source.examples(), limit)
        ]
        return await self._fetch_examples(synced_name, example_ids)

    async def _fetch_examples(
        self, dataset_name: str, example_ids: list[uuid.UUID]
    ) -> list[schemas.Example]:
        """Return the dataset's examples with example_ids, in that order."""
        fetched: dict[uuid.UUID, schemas.Example] = {}
        batches = itertools.batched(
            example_ids, eval_settings.langsmith.ids_per_request
        )
        for batch in batches:
            listed = await asyncio.to_thread(
                list,
                self._client.list_examples(
                    dataset_name=dataset_name, example_ids=batch
                ),
            )
            fetched.update((example.id, example) for example in listed)
        return [fetched[example_id] for example_id in example_ids]


__all__ = [
    "ExperimentConfig",
    "ExperimentMetadata",
    "ExperimentReport",
    "ExperimentRunner",
    "Target",
]
