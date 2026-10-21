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

A2A does not standardize tool calls, prompt injection or seeded history, so
the harness uses this small convention on top of it. An agent that follows it
gets every grader and every command.

**Request.** Each user message is one A2A message with one text part.
`message.metadata` holds:

- `prompt_overrides`: `{prompt name: text}`. Use the text in place of your own
  prompt of that name, for this message only. Honor it in dev and test only.
- `history`: earlier messages in OpenAI chat format. It is sent only on the
  first message of a new context; seed the conversation with it.
- Every other key is the case's `context`, such as `channel`.

The first message omits `context_id`. Reply with one, and the harness sends
it back on every later message of the conversation. An `INPUT_REQUIRED` task
is continued with the same task id.

**Reply.** Complete the task with:

- an artifact named `trajectory` holding one data part,
  `{"messages": [...]}`. These are this turn's agent messages in OpenAI
  format: each assistant message with `tool_calls` (`function.arguments` a
  JSON string), each tool result as `{"role": "tool", "tool_call_id": ...,
  "content": ...}`, then the final assistant message;
- the final reply text in the status message, or in another text artifact;
- `applied` in that message's metadata: a list naming each of `history` and
  `prompt_overrides` that you were sent and used.

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
