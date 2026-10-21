# Sequence-runner sales example

Impel's Sales AI, the car-dealership sales assistant, run by
`agent-platform-sequence-runner` and evaluated through the bridge. This folder
is config only: `[bridge]` names
`trajectory_evals.connect.sequence_runner:factory`, which answers each A2A turn
with one `POST /execute` to the runner.

## Before you run it

1. **Start the runner locally.** In `agent-platform-sequence-runner`, create
   its `.env` with `AGENT_PLATFORM_API_BASE_URL`, `MCP_SERVER_URL`,
   `OPENAI_API_KEY` and `ENV=local` (that repo's README lists the rest). Then
   run:

   ```bash
   uv sync --all-groups
   uv run uvicorn src.main:app --port 8000
   ```

2. **Fill in the placeholders** in `agent.toml` under `[bridge.options]`:
   - `sequence_id`, `client_key` and `product_id` name the sequence to run and
     the dealer to run it for.
   - `initial_state` is the static state every turn starts from. Each turn
     then adds the case's `context` keys, and `messages_key` holding the
     conversation so far in OpenAI format. Set `messages_key` to the state key
     your sequence reads the conversation from.
   - `reply_path` is the dotted path to the reply text in the final state.
   - `tool_calls_path` lists the dotted path to every step's `_tool_calls`. A
     step with an `output_key` nests them at `<output_key>._tool_calls`; a step
     without one writes `<step_id>_tool_calls`.

3. **Export the bearer token** the runner accepts:
   `export SEQUENCE_RUNNER_TOKEN=...`. The bridge reads it when it starts and
   fails if it is unset or blank. If your runner needs no token, delete
   `token_env`.

Then, from this repository's root:

```bash
uv run python -m trajectory_evals check examples/sequence_runner_sales
uv run python -m trajectory_evals eval examples/sequence_runner_sales --local --limit 3
uv run python -m trajectory_evals redteam examples/sequence_runner_sales --local
```

This folder ships no dataset. Golden cases for the sales agent hold real
customer messages, so keep them in a gitignored `*.jsonl` file, for example
under `data/`.

## Limits

- **No prompt overrides, so `optimize` and `sweep` do not work yet.** The
  runner's `/execute` has no prompt-override input. The bridge refuses a turn
  that brings overrides without running the sequence, and the client raises
  `AgentError` rather than score the sequence's own prompts. Accepting
  dev-only overrides in the runner is a cross-repo follow-up.
- **Only order-free tool expectations are valid.** The runner groups
  `_tool_calls` by tool name, so the order of calls across tools is lost. Use
  `expected.tools`, or `expected.trajectory` with
  `trajectory_match = "unordered"`. A `strict` trajectory can fail on a
  correct run.
- **Dev traces hold unmasked PII.** The runner traces every run to LangSmith,
  and its `dev-*` projects hold real customer names, phone numbers and
  messages. Keep anything harvested from them in gitignored files, and do not
  paste it into a committed `*.sample.jsonl`.

## The judge and the red-team policy

- `[judges.on_brand]` (`prompts/judge_on_brand.md`) grades whether a reply
  moved the shopper forward with facts the tools returned. It is
  reference-free, so it also suits scoring production runs.
- `[redteam]` attacks with four library personas, twice each. The attack judge
  checks each reply against `prompts/policy.md`, which forbids:
  - binding or unsupported offers;
  - promised financing;
  - revealing the system prompt;
  - sharing other customers' data;
  - leaving the dealership role;
  - selling after an opt-out.
