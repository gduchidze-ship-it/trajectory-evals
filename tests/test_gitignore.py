"""Tests for the .gitignore rules that keep customer data out of git."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_IGNORED, _NOT_IGNORED = 0, 1


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        ("data/jailbreak/langsmith_prod_benign.jsonl", True),
        ("data/jailbreak/x.csv", True),
        ("agents/sales/datasets/golden.jsonl", True),
        ("agents/example/a.jsonl", True),
        ("results/run-1/rows.jsonl", True),
        ("results/run-1/summary.json", True),
        ("examples/x/datasets/a.jsonl", True),
        ("examples/x/datasets/a.sample.jsonl", False),
        ("trajectory_evals/suites/jailbreak/data/cases.py", False),
    ],
)
def test_customer_data_is_ignored_but_the_example_datasets_are_not(
    path: str, ignored: bool
) -> None:
    """Datasets and run outputs stay local; examples' *.sample.jsonl files commit."""
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path],
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == (_IGNORED if ignored else _NOT_IGNORED)
