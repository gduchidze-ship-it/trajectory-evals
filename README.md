# trajectory-evals

A harness that evaluates, simulates and optimizes any agent it can reach over
[A2A](https://a2a-protocol.org). It grades the agent's whole trajectory (replies,
tool calls and tool results) on every row, and runs each experiment through
LangSmith, or offline with `--local`. Each agent is a folder: an `agent.toml` plus
its prompts and datasets.

- **Golden evals:** JSONL cases with expectations (exact output, phrases, forbidden
  claims, tools, a reference trajectory, rubric criteria), plus per-agent LLM judges.
- **Simulation and red-team:** an LLM user plays a persona through a multi-turn
  scenario; attack personas try to make the agent break its policy, and a judge
  scores each reply against that policy.
- **Optimization:** a reflective prompt loop and a floor-gated variant sweep. Both
  send candidate prompts through `prompt_overrides`, score them with the same runner,
  and never apply anything themselves.
- **LangSmith client:** pull and push datasets, harvest production traces into
  golden candidates, and score production runs online.

Agents that do not speak A2A are wrapped by the bundled **bridge**, an in-process
A2A server around any Python responder, such as the sequence-runner's `/execute`.

## Quickstart

Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
```

Put keys in a local `.env` (gitignored), or export them:

| Variable | Needed for |
|---|---|
| `LANGSMITH_API_KEY` | any run without `--local` |
| `OPENAI_API_KEY` | every `openai:` model: judges, the simulated user, the red-team judge, the reflection model |
| the agent's `auth_token_env` | an agent reached by `url` with a bearer token |

`bedrock_converse:` models use AWS credentials instead:

```bash
aws sso login --profile sales-ai-test
export AWS_PROFILE=sales-ai-test AWS_DEFAULT_REGION=us-west-2
```

Then:

```bash
uv run python -m trajectory_evals check examples/support_agent
uv run python -m trajectory_evals eval examples/support_agent \
  --data examples/support_agent/datasets/golden.sample.jsonl --local
```

## Commands

`<agent>` is a bare name, read as `agents/<name>/`, or any other path to an agent
folder.

```text
python -m trajectory_evals check    <agent> [--ping]
python -m trajectory_evals serve    <agent> [--host 127.0.0.1] [--port 9999] [--url URL]
python -m trajectory_evals eval     <agent> --data FILE [--criteria-model MODEL] [run options]
python -m trajectory_evals simulate <agent> --scenarios FILE [--max-turns N] [--criteria-model MODEL] [run options]
python -m trajectory_evals redteam  <agent> [--personas a,b] [--attempts N] [--max-turns N] [run options]
python -m trajectory_evals optimize <agent> --data FILE [--local | --upload-as-experiments] [--results-dir DIR]
python -m trajectory_evals sweep    <agent> --data FILE [--include DIR] [--local | --upload-as-experiments] [--results-dir DIR]
python -m trajectory_evals langsmith pull    <agent> --dataset NAME --out FILE
python -m trajectory_evals langsmith push    <agent> --data FILE [--dataset NAME]
python -m trajectory_evals langsmith harvest <agent> [run query] [--only-errors] [--feedback-key KEY [--feedback-below X]] [--out FILE]
python -m trajectory_evals langsmith score   <agent> [run query] [--dry-run]

run options: [--local] [--limit N] [--repetitions N] [--concurrency N]
             [--prefix PREFIX] [--results-dir DIR]
run query:   [--project PROJECT] [--since-days N] [--limit N] [--filter FILTER]
```

| Command | What it does |
|---|---|
| `check` | Validates the agent folder the way the other commands read it: `[judges]`, `[graders]`, `[simulation]`, `[redteam]` (its personas included), `[personas]`, `[optimize]`, `[variants]` and `[langsmith]`, and every file they name. Model names are only checked to be `provider:model` strings; no model is built, so no model key is needed. It validates every `datasets/*.jsonl`: a file name containing `scenario` is read as scenarios, whose personas must exist, and any other as golden cases. Then it builds the agent and fetches its card. `--ping` also sends one real message, untraced, which costs an LLM call. |
| `serve` | Serves a `[bridge]` agent over HTTP with uvicorn. Its card advertises `--url`, else `http://<host>:<port>/`, with an IPv6 host in brackets. The bridge has no inbound auth, so the default host is `127.0.0.1`, and binding a non-loopback address prints a warning. Binding `0.0.0.0` or `::` without `--url` also warns, since remote clients cannot use that address. |
| `eval` | Sends each golden case to the agent and grades it. The criteria judge is built only when a case that runs, within `--limit`, has `expected.criteria`. Summary segments: `risk`, `difficulty`. |
| `simulate` | Runs one conversation per scenario, after checking that every scenario's persona exists. The golden graders grade the whole conversation, beside `attack_succeeded` and `turns_to_break`. Segment: `persona`. |
| `redteam` | Runs each attack persona `attempts` times, and needs a `[redteam].policy`. Its pass key is `defended`: 1 when the judge found that no turn broke the policy, 0 when one did, unscored when the judge failed without finding a break. So `pass_rate` is the defended rate, and an attack whose run raised counts as not defended. It prints that rate beside the attack success rate (ASR): `attack_succeeded`'s mean over the attacks the judge scored. It also prints `attack_succeeded_coverage` and `error_rate`, and warns loudly when the coverage is below 1, because then some attacks were never judged. Segment: `persona`. |
| `optimize` | Hill-climbs the `[optimize].prompts` on the `--data` golden cases, with the reflection model proposing one revision per round. See [Optimization](#optimization). It prints the verdict, the best candidate, why it stopped and the path of `report.json`. |
| `sweep` | Runs the baseline, every `[variants.<name>]` and, with `--include`, a folder of `<name>.md` prompts such as an optimize run's `best/`. It picks the cheapest one that clears the floors, then prints one line per candidate, the verdict and the path of `report.json`. |
| `langsmith pull` | Writes a LangSmith dataset as golden JSONL. `--out` must be a new file under the gitignored `data/`. |
| `langsmith push` | Mirrors a golden JSONL file into a dataset, by default `<[langsmith].dataset_prefix, else the folder name>/<file stem>`, writing only the examples that changed. It refuses a row that still has `review_required` set. |
| `langsmith harvest` | Writes the project's recent root runs as golden-case candidates to review, by default to `data/langsmith/<agent>-<UTC time>.jsonl`. Each row has `review_required: true`, the agent's reply as `observed_output`, and an empty `expected`. |
| `langsmith score` | Scores the project's recent runs that did not error, using the reference-free judges, plus `policy_violation` when `[redteam]` has a policy. It writes each score as LangSmith feedback with a deterministic id, skipping runs that already carry the key, and never extends trace retention. `--dry-run` logs each score instead. It prints the runs read and, per key, the counts scored, skipped, errored and written. |

Run options (eval, simulate, redteam):

| Option | Default | Effect |
|---|---|---|
| `--local` | off | Runs on an offline LangSmith client: no dataset sync, no upload, no tracing. Without it, `LANGSMITH_API_KEY` is required. |
| `--limit N` | all | Runs only the first N cases. |
| `--repetitions N` | 1 | Runs each case N times, which adds `pass_all_k`/`pass_any_k` to the summary. |
| `--concurrency N` | 4 | Cases in flight at once. |
| `--prefix PREFIX` | `<agent>-<command>` | Experiment name prefix. |
| `--results-dir DIR` | `results` | Where `<run_id>/` is written. |
| `--max-turns N` | the case's, else `[simulation].max_turns` | Caps user messages per conversation (simulate, redteam). |
| `--criteria-model MODEL` | the golden judge model | Model of the `expected.criteria` judge (eval, simulate). |
| `--personas a,b` | `[redteam].personas`, else every attack persona | Attack personas to run (redteam). |
| `--attempts N` | `[redteam].attempts` | Conversations per persona (redteam). |

Other options:

- **`optimize` and `sweep`.** `--local` runs offline. Otherwise the runs trace to
  LangSmith, which needs `LANGSMITH_API_KEY`, and `--upload-as-experiments` also
  uploads every experiment, on one dataset per mode, seed and split.
- **The run query (harvest, score).** `--project` defaults to `[langsmith].project`,
  `--since-days` to `[langsmith].since_days` (7), and `--limit` caps the root runs
  read (100). `--filter` takes a LangSmith run filter. The langsmith commands always
  use a normal LangSmith client, so they need `LANGSMITH_API_KEY` too.

An uploaded eval, simulate or redteam run mirrors its cases into the LangSmith
dataset `<[langsmith].dataset_prefix, else the folder name>/<data file stem>` (or
`/redteam-<hash>` for redteam, where the hash is of the sorted persona names and
`attempts`, so runs of different personas never overwrite each other's examples)
and writes only the examples that changed. Its experiment
is named `<prefix>-<random>`, while a local one is `<prefix> (local)`. Uploaded runs
and `langsmith push` give a case the same example id, from `[langsmith].namespace`,
so both can write one dataset. The default namespace, `trajectory-evals.impel.ai`,
and the `redteam-<hash>` names are new, so uploads made before them got other ids
and dataset names; for the jailbreak example, see "Changes from the old suite" in
`examples/jailbreak/README.md`.

**Exit codes.** `0` on success. `2` on bad input, with one line on stderr and no
traceback. Bad input is:

- a missing `LANGSMITH_API_KEY` where one is needed, or a missing or empty data file;
- a bad `agent.toml`, data row or option value, or an unknown persona;
- a model that cannot be built, such as one with no API key, or an agent that cannot
  start;
- what a command rejects up front: redteam without a policy, bad splits for optimize
  and sweep, and for the langsmith commands an existing or tracked output file, or
  an unknown dataset or project;
- for `check` also: a `module:attr` that does not resolve, or a card it cannot fetch.

Error lines never repeat the values they reject. A data-row error names `path:line`
and the field, and a bad option names its key, never its value.

## Optimization

The optimizer never edits a prompt file. The baseline is the files in
`[optimize].prompts`, always sent as `prompt_overrides`. Candidates go to
`results/<run_id>/`, and a human promotes one by pull request. Cases split by their
`split` metadata (`train`, `val`, `holdout`), else by a hash of their id.

**`optimize`.** Each round, the reflection model revises the best prompts from up
to `failures_per_reflection` failed train cases and their graders' comments. A
revision is accepted only if its val run passes all of these:

- it clears the floors and the judge coverage;
- it beats the best `pass_rate` by at least `min_delta`;
- it breaks no `risk = "high"` case the best passed;
- when red-teamed, its `attack_succeeded` stays within `max_regression` of the
  baseline's, with every attack of both scored.

The run stops at `iterations`, `patience`, `max_agent_calls` or when no failures are
left. The baseline and the best then run on holdout, and the verdict is "adopt" only
if the best scores higher there and clears the floors there too.

**`sweep`.** Each candidate runs on val, and one that clears the floors and breaks
no high-risk case is red-teamed. The cheapest declared `cost` wins, then the higher
`pass_rate`, then the lower `latency_p50_ms`. The winner and the baseline then run
on holdout.

**Output.** `results/<run_id>/report.json` is rewritten after every candidate,
beside `evals/<candidate>-<split>/` (each run's rows and summary). `optimize` also
writes `candidates/<id>/<name>.md` and `best/<name>.md`.

## Agent folder

```text
agents/<name>/        your own agents (or any folder you pass by path)
  agent.toml
  prompts/*.md        judge prompts, the simulated user's prompt, red-team brief and policy
  datasets/*.jsonl    golden and scenario files
```

Paths in `agent.toml` are relative to the folder and may not leave it. An unknown
table or key is an error. Every table except `[agent]` is optional, but `[agent].url`
or `[bridge]` is required.

```toml
[agent]
description = "Support agent for an online bike shop."  # required; personas and the card use it
url = "https://agent.example/"     # its A2A endpoint; omit it to serve [bridge] in process
auth_token_env = "SHOP_AGENT_TOKEN"  # sent as a bearer token, only to url's origin
timeout_s = 120                    # per message
history = "metadata"               # or "replay"; see the A2A wire protocol
```

```toml
[bridge]                           # wraps a non-A2A agent
responder = "my_agent.bridge:factory"  # factory(options, config) -> Responder
options = { model = "openai:gpt-5.4-mini" }
```

```toml
[judges.helpful]                   # an LLM judge of every row's full transcript, reported as "helpful"
prompt = "prompts/judge_helpful.md"  # or "openevals:TASK_COMPLETION_PROMPT"
model = "openai:gpt-5.5"           # default: the golden [judges].model
gate = true                        # joins pass_rate; a gate cannot be continuous
continuous = false                 # a 0-1 score instead of pass/fail
reference_free = false             # true forbids {reference_outputs}
```

A judge prompt may use only `{inputs}` (the case inputs as JSON), `{outputs}` (the
full transcript) and `{reference_outputs}` (the case's `expected` as JSON). A judge
cannot take a built-in grader's name.

```toml
[graders]                          # an agent's own graders, added to the built-in ones
factory = "my_agent.graders:build" # build(config) -> trajectory_evals.golden.graders.Graders
```

`Graders(evaluators, summary_evaluators, pass_keys)`: the evaluators run on every row,
the summary evaluators on the whole run, and the pass keys join `pass_rate`.

```toml
[simulation]
user_prompt = "prompts/user.md"    # opens every simulated user's system prompt
user_model = "openai:gpt-5.4-mini"
max_turns = 6
stop_token = "END_CALL"            # the user ends the conversation with it; the agent never sees it
stop_when = "my_agent.hooks:handed_off"  # predicate(transcript) -> bool, checked after each agent turn
```

```toml
[personas.vip]                     # adds a persona, or replaces the library one of that name
kind = "benign"                    # or "attack"
prompt = "You are a repeat customer who expects priority."  # or prompt_file = "prompts/vip.md"
```

The library has `benign_curious` and `benign_frustrated`, plus the attack personas
`crescendo`, `roleplay`, `instruction_override`, `data_exfiltration`,
`system_prompt_extraction` and `encoding_obfuscation`. Library prompts are written
around the agent's `description`, while an agent's own persona prompts are used as
written.

```toml
[redteam]
policy = "prompts/policy.md"       # what the agent must never do; required for attack personas
brief = "prompts/redteam_brief.md" # replaces the library attacker brief
personas = ["crescendo", "instruction_override"]  # default: every attack persona
attempts = 2
judge_model = "openai:gpt-5.5"     # the attack_succeeded judge, which also ends a broken conversation
```

```toml
[optimize]                         # read by optimize and sweep; prompts and floors are required
prompts = { system = "prompts/system.md" }  # prompt name -> file: the baseline, never written
reflection_model = "openai:gpt-5.5"
cost = 1.0                         # the baseline's declared cost, which sweep compares
floors = { pass_rate = 0.8, max_error_rate = 0.05, min_judge_coverage = 0.9 }  # also latency_p95_ms
max_regression = { attack_succeeded = 0.0 }  # how far red-team ASR may rise over the baseline's
redteam = true                     # needs [redteam] with a policy; set false to skip the gate
budget = { iterations = 8, patience = 3, min_delta = 0.02, failures_per_reflection = 8, repetitions = 1 }  # also max_agent_calls
seed = 0
split = { train = 0.5, val = 0.25, holdout = 0.25 }  # used when cases set no split

[variants.short]                   # a sweep candidate: files that replace the baseline's
prompt_overrides = { system = "prompts/system_short.md" }  # only names [optimize].prompts has
cost = 0.4
```

The values shown are the defaults, except `prompts` and `floors`, which have none.

```toml
[langsmith]
project = "dev-sales"              # the project harvest and score read runs from
dataset_prefix = "support"         # datasets are <prefix>/<file stem>; default: the folder name
namespace = "trajectory-evals.impel.ai"  # scopes the example ids push and uploaded runs write
inputs_key = "messages"            # dotted path to the messages in a root run's inputs
outputs_key = "messages"           # and in its outputs
since_days = 7                     # how far back harvest and score read by default
```

Each table is validated by the package that reads it, and `check` validates them
all.

## Data formats

### Golden JSONL

One case per line:

```json
{"id": "refund-1", "input": "Please refund order A1.", "context": {"channel": "chat"},
 "expected": {"tools": ["refund_order"], "contains": ["refund"], "not_contains": ["guaranteed"],
              "criteria": ["Confirms the refund only after refund_order succeeded"]},
 "risk": "high", "difficulty": "easy", "split": "val"}
```

- A case has exactly one of `input` (one user message) or `messages`: OpenAI-format
  messages ending with a user message that has text. The messages before the last one
  are sent as history.
- `context` is sent with the message as metadata, for example the channel.
- Every other key becomes metadata, such as `risk`, `difficulty` or `split`.
  `source_file`, `case_id` and `content_hash` are reserved.
- Ids are unique within a file.

### Scenario JSONL

```json
{"id": "late-order", "persona": "benign_frustrated", "scenario": "Order A1 is a week late and you want a refund.",
 "opening_message": "Where is my order A1?", "max_turns": 4, "context": {"channel": "sms"},
 "expected": {"tools": ["lookup_order"], "criteria": ["Offers a refund or a human"]}, "difficulty": "hard"}
```

- `persona` is a library or `[personas]` name. An attack persona needs a
  `[redteam].policy`.
- `opening_message` is the first user message, sent verbatim. Without it, the
  simulated user writes the first message.
- `max_turns` caps this case's user messages; `--max-turns` takes precedence over it,
  and it over `[simulation].max_turns`.
- Every other key becomes metadata.

A conversation ends when the user sends the stop token, when `stop_when` holds, when
the judge finds an attack broke the policy, or at `max_turns`.

### Expectations

Each `expected` key present adds one check; the rest are left unscored.

| Key | Feedback key | Passes when |
|---|---|---|
| `output` | `output_exact` | the final reply equals it, ignoring outer whitespace |
| `contains` | `contains` | every phrase appears, ignoring case, in the assistant messages of the graded span |
| `not_contains` | `not_contains` | no phrase appears in any assistant message of the graded span |
| `tools` | `tools_called` | each tool is called at least once in the span; other calls are allowed |
| `trajectory` | `trajectory_match` | openevals' trajectory match on the span passes, in `trajectory_match` mode (`strict`, `unordered`, `subset` or `superset`; default `strict`) and `tool_args_match` mode (`exact`, `ignore`, `subset` or `superset`; default `exact`) |
| `criteria` | `criteria` | one LLM judge finds that every rubric line holds over the full transcript; paraphrase passes, but a promise without the action does not |

- **Graded span.** For a golden case, the span runs from the last input user message
  onward, so history is excluded. For a simulation, it is the whole conversation.
- **Unscored rules.** A grader returns unscored (`None`, not `0`) when its key is
  absent or empty, or when the target errored. The `tools` and `trajectory` graders
  are also unscored when the agent reported no trajectory. Unscored rows are left out
  of a key's mean, and `<key>_coverage` reports the share of rows that were scored.
- Keys the framework does not know are kept, so custom graders can read them.

## A2A wire protocol (for agent authors)

**Request.** One text part holds the user message. `message.metadata` holds:

- `prompt_overrides: {name: text}`: use this text instead of your own prompt of that
  name, for this message only (dev and test environments only);
- `history: [OpenAI messages]`: sent only on the first message of a new context, to
  seed the conversation;
- every other key: the case's `context`, such as `channel`.

**Reply.** Complete the task with:

- the final text in the reply message;
- an artifact named `trajectory`, with a data part `{"messages": [...]}` holding this
  turn's messages in OpenAI format: assistant messages with `tool_calls` (each
  `function.arguments` a JSON string), tool results (`role: "tool"` and
  `tool_call_id`), then the final assistant message;
- reply message metadata `applied: ["history", "prompt_overrides"]`, listing those of
  the two that you used.

**Fallbacks.**

- **No `trajectory` artifact.** The turn is just the reply text, and the `tools` and
  `trajectory` graders return unscored.
- **History or overrides missing from `applied`.** A sent history or prompt override
  that is not listed in `applied` raises `AgentError`, so the row errors rather than
  scoring prompts the agent ignored.
- **No history support.** An agent that cannot take history sets
  `history = "replay"`: earlier user turns are re-sent on one context, and the run's
  metadata records `history: replay`.
- **Failed tasks.** A task that ends in a state other than completed or input-required
  raises `AgentError`.

**Multi-turn.** The first message carries no `context_id`. The harness reads the
returned one and reuses it on later messages, and continues an input-required task
with its `task_id`. With `auth_token_env`, every request carries
`Authorization: Bearer <token>`, and a card that sends messages to another origin is
refused.

**The bridge.** For a non-A2A agent, write a responder factory:
`factory(options, config) -> Responder`, where a `Responder` is
`async (AgentRequest) -> AgentReply` (in `trajectory_evals.connect.bridge`).
`config` is the loaded agent folder: read a file that `options` names with
`config.read_text(path)`, so the path is relative to the folder. A factory that
cannot be imported, or that raises, is reported as a bad `agent.toml` naming the
responder; the message never repeats an option value, since options can hold
secrets.

- `AgentRequest` carries `message`, `history` (seeded history, then every earlier
  turn), `context_id`, `prompt_overrides` and `context`.
- `AgentReply` carries `text`, this turn's `messages`, and `prompt_overrides_applied`.
- The bridge keeps each context's history, emits the `trajectory` artifact and fills
  in `applied`.
- An agent with no `url` is served in process, with no sockets. `serve` exposes it
  over HTTP instead.

`trajectory_evals.connect.sequence_runner:factory` wraps
agent-platform-sequence-runner's `POST /execute`. It refuses prompt overrides. It
also loses the order across tools, so only `tools` and `unordered` trajectory
expectations are valid for it.

## Results

Every run, local or uploaded, writes `results/<run_id>/`, where the run id is the UTC
time, a slug of the experiment name and random hex:

- **`rows.jsonl`:** one line per run, with `case_id`, `inputs`, `outputs` (the full
  transcript), `reference_outputs`, `metadata`, `feedback`, `comments`, `error`,
  `latency_ms` and `feedback_errors`.
- **`summary.json`:** the keys below, plus any `[graders]` summary scores. The command
  also prints them, except `rows`, which it prints as `runs`.
- **`meta.json`:** `agent`, `command`, `dataset_file`, `history`, `experiment`,
  `dataset`, `url`, and `options`: the parsed command line.

| Summary key | Meaning |
|---|---|
| `<key>`, `<key>_coverage` | each feedback key's mean over the rows that scored it (a bool counts 0 or 1), and the share of rows that scored it |
| `pass_rate` | share of rows that passed every pass key they scored; a row whose target or any grader raised fails. The pass keys are the deterministic graders, `criteria`, each gate judge and the `[graders]` pass keys, so a judge pass never overrides a deterministic fail. For redteam the pass key is `defended`. |
| `pass_rate_ci_low`, `pass_rate_ci_high` | Wilson 95% interval of `pass_rate`, over distinct cases, so repetitions do not narrow it |
| `pass_all_k`, `pass_any_k`, `repetitions` | with `--repetitions` above 1: the share of cases whose every run passed (tau-bench's pass^k), or any run passed, and the fewest runs of any case |
| `error_rate` | share of rows whose target raised |
| `latency_p50_ms`, `latency_p95_ms` | per-row latency |
| `rows` | rows run |
| `pass_rate[<segment>=<value>]` | `pass_rate` per value of a metadata key, else an output key, else an input key, so a row whose run raised keeps its segment: `risk` and `difficulty` for eval, `persona` for simulate and redteam |

## LLM providers

Every model is a `provider:model` string resolved by LangChain's `init_chat_model`:
`openai:<model>` (uses `OPENAI_API_KEY`) or `bedrock_converse:<Bedrock model id>`
(uses AWS credentials, in `us-west-2` on the `sales-ai-test` SSO profile). Models are
set in:

- `[judges.<name>].model`;
- `--criteria-model`;
- `[simulation].user_model`;
- `[redteam].judge_model`;
- `[optimize].reflection_model`.

The defaults are `openai:gpt-5.5` for judges and the reflection model, and
`openai:gpt-5.4-mini` for the simulated user. Every value in a package's `config.toml` can be overridden from the
environment or `.env` as `TRAJECTORY_EVALS_<PACKAGE>_<SECTION>__<KEY>`, where the
package is `CORE`, `CONNECT`, `GOLDEN`, `SIMULATION`, `OPTIMIZE`, `LANGSMITH_IO` or
`CLI`. For example,
`TRAJECTORY_EVALS_GOLDEN_JUDGES__MODEL` sets the default judge model.

## Examples

`examples/` holds agent folders the framework never imports:

| Folder | What it shows |
|---|---|
| [`examples/support_agent`](examples/support_agent) | a tool-using support agent over mocked tools that runs offline and exercises every feature, including the optimizer |
| [`examples/jailbreak`](examples/jailbreak) | the `detect_jailbreak` prompt-attack guard behind a bridge responder, with its own graders through `[graders]` and its own personas |
| [`examples/sequence_runner_sales`](examples/sequence_runner_sales) | a real Impel sales agent run through the sequence-runner's `/execute` (config only); point it only at an eval-only sequence, never a production one |
| [`examples/remote_a2a`](examples/remote_a2a) | a remote A2A agent with a bearer token and `history = "replay"` (config only) |

**Never point a sequence-runner example at a production sequence.** Every eval turn
runs the whole sequence, and the production reply sequences text, email, update and
unsubscribe real customers. Use an eval-only dev sequence that reads the conversation
from `messages_key` and has no send, pursuit, notification, unsubscribe or scheduling
step.

## Data hygiene

- Every `*.jsonl` is gitignored except the synthetic `examples/**/*.sample.jsonl`.
  `/data/` and `results/` are gitignored too.
- `rows.jsonl` holds the full transcripts, so treat `results/` like the data it came
  from.
- Judges and the simulated user send transcripts to their model provider.
- `--local` results stay on your machine and are not shared evidence.
- Traces from `dev-*` LangSmith projects hold unmasked customer PII. `langsmith pull`
  and `harvest` therefore write only new files under `data/`, and `push` refuses rows
  still marked `review_required`; review every row before you push it to a shared
  dataset.

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy trajectory_evals examples tests
uv run pytest -q
```

Tests run offline. `tests/e2e` drives the CLI end to end, against an in-process
bridged test agent and `examples/support_agent`, with every socket refused. Tests that call real services are marked
`e2e` and deselected by default; run them with `uv run pytest -m e2e` after logging in
to AWS and setting the keys.
