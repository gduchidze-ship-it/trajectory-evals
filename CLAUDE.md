# CLAUDE.md

Eval and simulation framework for Impel AI agents over A2A. Layout and usage: `README.md`.

## Commands

```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check . && uv run mypy trajectory_evals examples tests
uv run pytest -q                      # e2e is deselected; `-m e2e` needs AWS_PROFILE=sales-ai-test
uv run python -m trajectory_evals check examples/support_agent
```

## Python style (ruff enforces D google, N, W, ERA; a hook runs it after every edit)

- One-line module docstring; Google-style docstring (Args/Returns/Raises) on every class, function, method.
- No inline or block comments. Put why-notes in a docstring `Note:` section. `# type: ignore` and `# noqa` stay.
- Pragmatic OOP: group related module-level functions into a class; tiny pure helpers are static methods.
- Pinned `==` versions in `pyproject.toml`.
- Every LLM role takes a `provider:model` string and works on OpenAI and Claude on Bedrock.

## Workflow

- Work lands on `refactor/a2a-harness`. A worktree branches from `main`, so start every worktree brief with `git reset --hard refactor/a2a-harness`.
- Large work: plan first, orchestrator session only, one isolated worktree agent per work package, independent review per package.
- Python edits go through the `code-writer` agent; run the `implementation-final-review` skill before calling it done. If either says to add comments, this file wins.
- Research an SDK's latest official docs (LangSmith, A2A, LangChain) before using it.

## Data

Customer messages under `data/` and any LangSmith pull stay on this machine. Never commit them or send them to a third-party service.
