"""The support agent's golden sample, run end to end offline through the runner."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from tests.examples.support_agent.support_fakes import (
    SUPPORT_DIR,
    RoutedChatModel,
    support_agent,
    support_config,
    tool_call,
)
from tests.fakes import judge_reply, scripted
from trajectory_evals.core.dataset import LangSmithDatasetSync
from trajectory_evals.core.runner import (
    ExperimentConfig,
    ExperimentRunner,
    build_local_client,
)
from trajectory_evals.core.summary import summarize
from trajectory_evals.golden.cases import JsonlGoldenSource
from trajectory_evals.golden.graders import build_graders
from trajectory_evals.golden.target import GoldenTarget

GOLDEN = SUPPORT_DIR / "datasets" / "golden.sample.jsonl"
ROWS = 3  # the sample's first rows, which SCRIPTS answers
KEYS = ("tools_called", "trajectory_match", "contains", "not_contains", "helpful")
SCRIPTS: dict[str, list[AIMessage]] = {
    "refund-eligible": [
        tool_call("lookup_order", "c1", order_id="A1001"),
        tool_call("refund_order", "c2", order_id="A1001"),
        AIMessage("Done: your refund of $89.00 arrives in 3-5 business days."),
    ],
    # A forbidden claim: the tool refused, but the reply says it refunded.
    "refund-outside-window": [
        tool_call("lookup_order", "c1", order_id="A1003"),
        AIMessage("No problem, I've refunded your stove."),
    ],
    "billing-dispute": [
        tool_call("escalate_to_human", "c1", reason="double charge on A1001"),
        AIMessage("I've escalated this as ticket T-5001; a person will email you."),
    ],
}


@pytest.mark.asyncio
async def test_the_golden_sample_runs_offline_and_grades_each_row(
    offline: list[Any],
) -> None:
    """Graders score the bridged trajectory; a lie fails even when the judge passes."""
    source = JsonlGoldenSource(GOLDEN)
    inputs = {case.id: case.input for case in source.cases()}
    model = RoutedChatModel(
        routes={str(inputs[case_id]): script for case_id, script in SCRIPTS.items()}
    )
    config = support_config()
    judge = scripted(*[judge_reply(True)] * ROWS)
    graders = build_graders(config, model_factory=lambda _: judge, criteria=False)
    client = build_local_client()
    runner = ExperimentRunner(
        client=client, sink=LangSmithDatasetSync(client, namespace="support")
    )

    async with support_agent(model) as agent:
        report = await runner.run(
            source=source,
            dataset_name="support/golden",
            target=GoldenTarget(agent),
            evaluators=graders.evaluators,
            summary_evaluators=graders.summary_evaluators,
            config=ExperimentConfig(
                experiment_prefix="support",
                max_concurrency=ROWS,
                upload_results=False,
                limit=ROWS,
            ),
            metadata={},
        )

    scores = {
        row.case_id: {key: row.feedback[key] for key in KEYS} for row in report.rows
    }
    assert scores == {
        "refund-eligible": {
            "tools_called": True,
            "trajectory_match": True,
            "contains": True,
            "not_contains": None,
            "helpful": True,
        },
        "refund-outside-window": {
            "tools_called": True,
            "trajectory_match": None,
            "contains": None,
            "not_contains": False,
            "helpful": True,
        },
        "billing-dispute": {
            "tools_called": True,
            "trajectory_match": None,
            "contains": True,
            "not_contains": True,
            "helpful": True,
        },
    }
    # helpful joins the AND as a gate judge would; its pass cannot rescue the lie.
    pass_keys = (*graders.pass_keys, "helpful")
    passed = {
        row.case_id: summarize([row], pass_keys=pass_keys)["pass_rate"]
        for row in report.rows
    }
    assert passed == {
        "refund-eligible": 1.0,
        "refund-outside-window": 0.0,
        "billing-dispute": 1.0,
    }
    summary = summarize(report.rows, pass_keys=pass_keys)
    assert summary["pass_rate"] == pytest.approx(2 / 3)
    assert offline == []
