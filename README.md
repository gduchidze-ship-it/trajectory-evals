# trajectory-evals

Eval framework for Impel AI agents. Each suite runs as a LangSmith experiment, or
in memory with `--local`.

## Suites

| Suite | What it measures | Docs |
|---|---|---|
| `evals/jailbreak` | The `detect_jailbreak` prompt-attack guard (Bedrock Guardrail, Haiku screen, Sonnet judge), on labelled single messages and on simulated multi-turn conversations | [evals/jailbreak/README.md](evals/jailbreak/README.md) |

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in the `eval`
group of `pyproject.toml`.

```bash
uv sync --group eval
aws sso login --profile sales-ai-test
export AWS_PROFILE=sales-ai-test AWS_DEFAULT_REGION=us-west-2
```

Put `LANGSMITH_API_KEY` and `OPENAI_API_KEY` in a local `.env`, or export them.

## Run

```bash
uv run --group eval python -m evals.jailbreak detection --limit 50 --local
uv run --group eval python -m evals.jailbreak simulation --sample 3 --max-turns 4 --local
```

Flags, data format, scores and configuration are in the suite README.

## Notes

- The detector under test, `src.toolsets.jailbreak_detection`, lives in
  `agent-platform-mcp-core-tools`, not here. It must be importable to run a suite.
- `data/` holds real customer messages. It is gitignored; never commit it.
- `--local` results stay on your machine and are not shared evidence.
