# Remote A2A agent example

An agent that already speaks A2A, reached over HTTP: `[agent] url` instead of
a `[bridge]`. This folder is config only. Point `url` at your agent's root,
where its agent card is served, and export its bearer token:

```bash
export REMOTE_AGENT_TOKEN=...
uv run python -m trajectory_evals check examples/remote_a2a
uv run python -m trajectory_evals redteam examples/remote_a2a --local
```

It has one judge, `[judges.task_completion]`, which uses openevals'
`TASK_COMPLETION_PROMPT` instead of a file in the folder. Its `[redteam]`
policy is `prompts/policy.md`.

## The wire convention, for agent authors

- **Request:** `message.metadata` carries `prompt_overrides` and, on a new
  context's first message, `history`; every other key is the case's context.
- **Reply:** a `trajectory` artifact with the data part `{"messages": [...]}`
  holding this turn's tool calls and results, and `applied` in the reply
  metadata naming what was used.

The full convention is in the wire protocol section of the
[root README](../../README.md).

## Agents that do not follow it

Third-party agents usually do none of this. The harness then degrades as
follows:

- **No trajectory artifact.** The turn becomes a single assistant message
  with the reply text. `tools_called` and `trajectory_match` return unscored,
  never a fail, and text graders and judges still run.
- **No history.** Set `history = "replay"`, as this folder does. The harness
  then sends a case's earlier user messages as real turns on one new context,
  before the message under test. The agent writes its own replies to them, so
  the conversation only approximates the recorded one.
- **No prompt overrides.** If an override is sent but not listed in
  `applied`, the client raises `AgentError` rather than score the agent's own
  prompts. `optimize` and `sweep` therefore fail loudly against such an agent.

## The bearer token

The client reads `auth_token_env` when it starts and fails if the variable is
unset or blank. It sends the token only to the origin of the configured
`url`. If the agent card lists an interface on any other origin, the client
raises `AgentError` before sending a message, so a card cannot redirect the
token elsewhere.
