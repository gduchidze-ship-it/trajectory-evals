"""Run one LangSmith experiment over a case source and report its rows and scores."""

import asyncio
import inspect
import itertools
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol

from langsmith import Client, EvaluationResult, aevaluate, schemas

from trajectory_evals.core.protocols import CaseSource, DatasetSink
from trajectory_evals.core.settings import settings

Target = Callable[[Any], Awaitable[Mapping[str, Any]]]


class _Experiment(Protocol):
    """The parts of aevaluate's results, a private langsmith class, naming reads."""

    @property
    def experiment_name(self) -> str:
        """Return the uploaded experiment's name."""

    async def get_comparison_url(self) -> str | None:
        """Return the URL comparing the experiment with others on its dataset."""


def build_local_client() -> Client:
    """Return a client with no tracing thread, so a local run sends nothing.

    It drops all feedback too, so never use it for an uploaded run.
    """
    return Client(auto_batch_tracing=False, tracing_sampling_rate=0.0)


def _format_score(score: float | None) -> str:
    """Return score to the configured decimals, or the undefined label if None."""
    report = settings.langsmith
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
            metadata={**case["metadata"], settings.data.case_id_key: case["case_id"]},
        )
        for case in itertools.islice(source.examples(), limit)
    ]


@dataclass(frozen=True)
class ExperimentRow:
    """One run of one case: what went in and out, its feedback, error and latency.

    case_id and metadata come from the example. outputs are the target's outputs
    exactly as returned; for a target that raised, LangSmith records
    {errored_output_key: None} and error holds the exception. feedback and
    comments map each evaluator's key to its score, None when unscored, and its
    comment. latency_ms is the run's end minus its start, None if it never ended.
    """

    case_id: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    reference_outputs: Mapping[str, Any]
    metadata: Mapping[str, Any]
    feedback: Mapping[str, float | bool | None]
    comments: Mapping[str, str | None]
    error: str | None
    latency_ms: float | None


def _experiment_row(
    run: schemas.Run, example: schemas.Example, results: Sequence[EvaluationResult]
) -> ExperimentRow:
    """Return one aevaluate row as the case, its outputs, feedback and timing."""
    metadata = example.metadata or {}
    end_time = run.end_time
    return ExperimentRow(
        case_id=metadata[settings.data.case_id_key],
        inputs=example.inputs or {},
        outputs=run.outputs or {},
        reference_outputs=example.outputs or {},
        metadata=metadata,
        feedback={result.key: result.score for result in results},
        comments={result.key: result.comment for result in results},
        error=run.error,
        latency_ms=None
        if end_time is None
        else (end_time - run.start_time) / timedelta(milliseconds=1),
    )


def _summary_scores(
    summary_evaluators: Sequence[Callable[..., Any]], rows: Sequence[ExperimentRow]
) -> dict[str, float | None]:
    """Return each summary score by key, computed on rows as aevaluate does."""
    arguments = {
        "inputs": [row.inputs for row in rows],
        "outputs": [row.outputs for row in rows],
        "reference_outputs": [row.reference_outputs for row in rows],
    }
    scores: dict[str, float | None] = {}
    for evaluator in summary_evaluators:
        # shortcut: takes only inputs/outputs/reference_outputs and returns dicts,
        # as every evaluator here does; add runs/examples when one needs them.
        parameters = inspect.signature(evaluator).parameters
        result = evaluator(**{name: arguments[name] for name in parameters})
        for item in result.get("results", [result]):
            score = item.get("score")
            scores[item["key"]] = None if score is None else float(score)
    return scores


@dataclass(frozen=True)
class ExperimentConfig:
    """How one experiment runs: name prefix, concurrency, repeats and upload mode."""

    experiment_prefix: str
    max_concurrency: int
    num_repetitions: int = settings.langsmith.repetitions
    upload_results: bool = True
    limit: int | None = None

    def __post_init__(self) -> None:
        """Reject a concurrency, repetition count or limit below one."""
        for name in ("max_concurrency", "num_repetitions", "limit"):
            value = getattr(self, name)
            if value is not None and value < 1:
                message = settings.texts.at_least_one
                raise ValueError(message.format(name=name, value=value))


@dataclass(frozen=True)
class ExperimentReport:
    """What one experiment produced: its name, dataset, URL, scores and runs.

    rows hold every run in completion order; equality and repr ignore them.
    """

    experiment_name: str
    dataset_name: str
    url: str | None
    summary: Mapping[str, float | None]
    runs: int
    rows: tuple[ExperimentRow, ...] = field(default=(), compare=False, repr=False)

    def render(self) -> str:
        """Return the report as the lines the CLI prints."""
        report = settings.langsmith
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
    results: _Experiment, config: ExperimentConfig
) -> tuple[str, str | None]:
    """Return the experiment's name and comparison URL, or a local label and no URL."""
    # Local results raise on experiment_id, url and get_comparison_url.
    if not config.upload_results:
        local_name = config.experiment_prefix + settings.langsmith.local_suffix
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
        """Evaluate target on the source's cases and return the experiment's report.

        summary_evaluators run inside aevaluate, so an upload shows them, and again
        here on the rows, called with whichever of inputs, outputs and
        reference_outputs they name and returning {"key", "score"} or
        {"results": [...]}. aevaluate only logs a summary evaluator that raises;
        here its exception propagates.
        """
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
        rows = tuple(
            [
                _experiment_row(
                    row["run"], row["example"], row["evaluation_results"]["results"]
                )
                async for row in results
            ]
        )
        experiment_name, url = await _name_experiment(results, config)
        return ExperimentReport(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            url=url,
            summary=_summary_scores(summary_evaluators, rows),
            runs=len(rows),
            rows=rows,
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
        batches = itertools.batched(example_ids, settings.langsmith.ids_per_request)
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
    "ExperimentReport",
    "ExperimentRow",
    "ExperimentRunner",
    "Target",
    "build_local_client",
]
