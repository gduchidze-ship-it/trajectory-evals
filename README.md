# trajectory-evals

Eval framework for Impel AI agents. Each suite runs as a LangSmith experiment, or
in memory with `--local`.

## Suites

| Suite | What it measures | Docs |
|---|---|---|
| `trajectory_evals/suites/jailbreak` | The `detect_jailbreak` prompt-attack guard (Bedrock Guardrail, Haiku screen, Sonnet judge), on labelled single messages and on simulated multi-turn conversations | [trajectory_evals/suites/jailbreak/README.md](trajectory_evals/suites/jailbreak/README.md) |

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in
`pyproject.toml`: the `eval` group runs the suites, `test` and `lint` check the code.

```bash
uv sync --all-groups
aws sso login --profile sales-ai-test
export AWS_PROFILE=sales-ai-test AWS_DEFAULT_REGION=us-west-2
```

Put `LANGSMITH_API_KEY` and `OPENAI_API_KEY` in a local `.env`, or export them.

## Run

```bash
uv run --group eval python -m trajectory_evals.suites.jailbreak detection --limit 50 --local
uv run --group eval python -m trajectory_evals.suites.jailbreak simulation --sample 3 --max-turns 4 --local
```

Flags, data format, scores and configuration are in the suite README.

## Check

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy trajectory_evals
uv run pytest -q
PYTHONPATH=../agent-platform-mcp-core-tools uv run pytest -q -rs tests/suites/jailbreak
```

The jailbreak suite's tests are skipped unless its detector is importable, as in the
last line.

## Notes

- The detector under test, `src.toolsets.jailbreak_detection`, lives in
  `agent-platform-mcp-core-tools`, not here. It must be importable to run a suite.
  Its Bedrock client needs `aioboto3`, which cannot install next to the pinned
  `boto3`, so the one test that builds the real detector skips here.
- `data/` and `results/` hold real customer messages. `results/` and every `*.jsonl`
  except under `agents/example/` are gitignored; never commit them.
- `--local` results stay on your machine and are not shared evidence.
