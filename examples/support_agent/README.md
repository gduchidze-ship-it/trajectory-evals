# Support agent example

The order support assistant of a made-up outdoor gear store. It is a LangChain
tool-calling loop over three mocked tools, wrapped by the bridge, and it is the
demo for every command: golden evals, simulations, red-teaming, the optimizer
and the variant sweep.

It needs no server. `agent.toml` has a `[bridge]` table and no `url`, so the
harness serves the responder in process over httpx's ASGI transport. Real runs
call `openai:gpt-5.4-mini` (the agent) and the judge, simulated-user and
reflection models, so they need `OPENAI_API_KEY`.

## Files

| Path | What it is |
|---|---|
| `agent.toml` | The agent folder's tables: `[bridge]`, `[simulation]`, `[judges.helpful]`, `[redteam]`, `[optimize]`, `[variants.short_prompt]` |
| `responder.py` | `factory(options, config, *, model_factory=init_chat_model)`: the tool loop and the mocked tools |
| `fixtures.json` | The mocks' canned results (four orders, their refund outcomes, the escalation ticket, the invalid-arguments result) and the handoff reply |
| `prompts/system.md` | The agent's system prompt, and the optimizer's baseline |
| `prompts/system_short.md` | A shorter system prompt, the `short_prompt` variant |
| `prompts/user.md` | Opens every simulated user's system prompt |
| `prompts/judge_helpful.md` | The `helpful` judge, run on every row; it reads only `{inputs}` and `{outputs}` |
| `prompts/redteam_brief.md`, `prompts/policy.md` | What the attackers are told, and the policy the attack judge enforces |
| `datasets/golden.sample.jsonl` | 12 golden cases |
| `datasets/scenarios.sample.jsonl` | 4 simulation scenarios, three benign and one hostile |

## Commands

Run them from the repository root. `--local` keeps results on disk under
`results/<run_id>/` and uploads nothing to LangSmith. Each command's `--help`
lists its flags.

```bash
uv run python -m trajectory_evals check examples/support_agent       # validate the folder and its card; --ping also sends one message
uv run python -m trajectory_evals serve examples/support_agent       # serve it as an A2A agent on localhost
uv run python -m trajectory_evals eval examples/support_agent --data examples/support_agent/datasets/golden.sample.jsonl --local --limit 3
uv run python -m trajectory_evals simulate examples/support_agent --scenarios examples/support_agent/datasets/scenarios.sample.jsonl --local --max-turns 2
uv run python -m trajectory_evals redteam examples/support_agent --local
uv run python -m trajectory_evals optimize examples/support_agent --data examples/support_agent/datasets/golden.sample.jsonl --local
uv run python -m trajectory_evals sweep examples/support_agent --data examples/support_agent/datasets/golden.sample.jsonl --local
```

What each one shows:

- **`check`** loads `agent.toml`, validates every table and the files it names,
  and parses the sample datasets.
- **`serve`** puts the bridge on a real port, so any A2A client can talk to it.
- **`eval`** runs the golden cases. The deterministic graders (`tools_called`,
  `trajectory_match`, `contains`, `not_contains`) need no model. The
  `criteria` judge and the `helpful` judge do. `pass_rate` ANDs the
  deterministic graders, the `criteria` judge and any judge with
  `gate = true` (`helpful` is not one), so a judge pass never hides a
  deterministic fail.
- **`simulate`** runs the scenarios. A simulated user plays each persona, its
  first turn is the scenario's `opening_message` when it has one, and the
  conversation ends at `END_CALL` or after `max_turns`.
- **`redteam`** attacks with the `crescendo`, `data_exfiltration` and
  `system_prompt_extraction` personas from the library, briefed by
  `prompts/redteam_brief.md`. The attack judge checks each reply against
  `prompts/policy.md` and stops the conversation at the first violation.
- **`optimize`** rewrites `prompts/system.md` from the failing train rows,
  sending each candidate as `prompt_overrides["system"]`. It keeps a candidate
  only if it clears the floors on val and does not raise the red-team attack
  success rate. It then compares the best candidate with the baseline on
  holdout. It never edits the prompt files; candidates go to
  `results/<run_id>/`.
- **`sweep`** evaluates the baseline and `[variants.short_prompt]` and picks
  the cheapest one that clears the floors.

## How the agent works

- **The loop.** The model is `model_factory(options["model"])` with the three
  tools bound on every round, made on the first turn, so `check` needs no API key. It may call tools for up to four rounds. If the
  fourth round still calls tools, the model is not called again: the turn ends
  with the fixed handoff reply in `fixtures.json`, so every turn ends with an
  answer.
- **The tools.** `lookup_order`, `refund_order` and `escalate_to_human` are
  plain functions that return canned results from `fixtures.json`, so the same
  arguments always give the same result. As in the OpenAI cookbook's tool
  mocks, the model's mistakes come back as results rather than errors: a call
  to any other tool gets `{"status": "Unsupported tool"}`, and wrong or
  missing arguments get `{"status": "Invalid arguments"}`.
- **The context.** The agent ignores a case's `context`; `channel` is carried
  for graders and segments only.
- **The reply.** Each turn returns its assistant tool calls, each tool result
  (with its `tool_call_id`) and the final answer, in OpenAI format with JSON
  string arguments. The bridge sends them as the `trajectory` artifact, which
  the trajectory graders read.
- **Prompt overrides.** `prompt_overrides["system"]` replaces
  `prompts/system.md` for that turn, and the reply acknowledges it. Overrides
  that name any other prompt are refused without calling the model, so the
  client raises `AgentError` instead of scoring the default prompt.
- **Providers.** Every model call has the tools bound, so the agent runs on
  OpenAI and on Claude through Bedrock Converse. Bedrock rejects a call with no
  tools once the history holds tool calls. To run on Bedrock, set
  `options = { model = "bedrock_converse:<model id>" }` under `[bridge]`.

## The golden rows

Rows follow the cookbook's `EVAL_SET` style:

- `expected_tools` becomes `expected.tools`.
- Required phrases become `expected.contains`.
- Forbidden claims, such as "I've refunded" when nothing was refunded,
  become `expected.not_contains`. The match is a plain substring that ignores
  negation, so these are only first-person action claims; "no refund has been
  processed" would trip a broader phrase.
- A per-case rubric becomes `expected.criteria`, which one judge grades as a
  single pass or fail. It covers what substrings cannot, such as a promised
  refund or a leaked prompt.

`refund-eligible` also has an `expected.trajectory` with
`trajectory_match = "superset"`: the agent must make those two tool calls,
with exactly those arguments, and may make others. `multi-turn-refund` gives
`messages` instead of `input`, and its earlier turns are sent as history.

The other keys are metadata. `intent`, `risk` and `difficulty` segment the
summary, for example `pass_rate[risk=high]`. `split` (`train`, `val` or
`holdout`) is the split the optimizer uses. Val mixes a case the agent should
complete (`refund-eligible`) with refusals, so a prompt that refuses too much
fails val.

## Tests

The tests run offline with no API key. They call
`factory(options, config, model_factory=...)` with a fake chat model that scripts the
tool calls, serve the responder with the bridge, and drive it through
`A2AAgent.in_process`. The golden test then runs the first rows of
`golden.sample.jsonl` through `GoldenTarget`, `build_graders` and the local
`ExperimentRunner`, with a scripted judge.

```bash
uv run pytest tests/examples
```

## Data

Every row and fixture here is synthetic. `.gitignore` ignores every `*.jsonl`
file except `examples/**/*.sample.jsonl`, and the top-level `data/`. Keep real
customer conversations in ignored files, and never name one `*.sample.jsonl`.
