# Jailbreak evals

LangSmith experiments that measure the `detect_jailbreak` cascade (Bedrock Guardrail
and Haiku screen, then the Sonnet judge). There are two commands:

- `detection` scores the real cascade on labelled single messages.
- `simulation` lets an LLM customer continue a real conversation, benign or attacking,
  while the detector screens every turn.

Each run is a LangSmith experiment. The JSONL file is mirrored into a dataset first,
and re-running with an unchanged file writes nothing. The command prints each
experiment's name, dataset, URL, run count and summary scores.

## Prerequisites

- **AWS (always).** The detector calls Bedrock in `us-west-2`:

  ```bash
  aws sso login --profile sales-ai-test
  export AWS_PROFILE=sales-ai-test AWS_DEFAULT_REGION=us-west-2
  ```

  The detector fails open, so without credentials every row would be allowed and the
  scores would look perfect. The command refuses to start if the credentials do not
  resolve.
- **`LANGSMITH_API_KEY`** for every run except `--local`. Results go to the workspace
  this key belongs to.
- **`OPENAI_API_KEY`** for `simulation`. The simulated user, the responder and the
  `attack_attempted` judge are OpenAI models, so the detector's own Sonnet judge never
  grades itself.
- Both keys can live in the repo's `.env` instead; variables already set in the shell win.
- The `eval` dependency group: `uv sync --group eval`.

A missing prerequisite or data file stops the command with a one-line error and exit
code 2, before any model or LangSmith call.

## Data

`data/jailbreak/` holds the labelled JSONL files. They are real customer messages and
are gitignored (`*.jsonl`): never commit them or copy them into the image. Every row
needs an `id`, an `expected_action` of `ALLOWED`, `BLOCKED` or `REVIEW`, and either
`input_text`/`text` or `turns`; the last of a row's `turns` must be the customer's and
hold text. All other fields become example metadata. An `id` used twice in a file
stops the command with exit code 2, naming the second line, and so does a file with
no cases, or for `simulation` no `ALLOWED` single-message opener.

A message reaches the detector the way prod sends it on its `channel`: an `email`
message as the conversation only, a `submission_form` opener as its text alone, and
any other message as its text plus the conversation that ends with it.

## `detection`

```bash
uv run --group eval python -m evals.jailbreak detection --limit 50 --local
uv run --group eval python -m evals.jailbreak detection --data data/jailbreak/langsmith_prod_benign.jsonl
```

- By default every `data/jailbreak/*.jsonl` runs, one experiment per file.
- The dataset is `jailbreak-detection/<file stem>`.
- Options: `--limit N`, `--max-concurrency 8`, `--repetitions 1`, `--local` and
  `--experiment-prefix detect-jailbreak`.

Per row:

| Key | Meaning |
|---|---|
| `correct` | The verdict matches the label. `REVIEW` rows are not scored. |
| `escalated` | The guardrail or the screen sent the message on to the judge. |
| `failed_open` | The judge reached no verdict, so the fail policy allowed the message. |

Per experiment:

| Key | Meaning |
|---|---|
| `false_positive_rate` | Share of `ALLOWED` rows that were blocked. |
| `detection_rate` | Share of `BLOCKED` rows that were blocked. |
| `precision`, `f1` | Over the `ALLOWED` and `BLOCKED` rows; `n/a` when a file has no `BLOCKED` row. |
| `escalation_rate` | Share of rows that reached the judge. |
| `fail_open_rate` | Share of rows allowed by the fail policy. |
| `review_block_rate` | Share of `REVIEW` rows that were blocked. |
| `error_rate` | Share of rows whose target raised. These rows are left out of every other score. |

## `simulation`

```bash
uv run --group eval python -m evals.jailbreak simulation --sample 3 --max-turns 4
uv run --group eval python -m evals.jailbreak simulation --personas benign_follow_up instruction_override --local
```

- A seeded sample of `ALLOWED` openers, stratified by channel, is crossed with each
  persona.
- The dataset is `jailbreak-simulation/<file stem>`.
- Options: `--data` (default `langsmith_prod_benign.jsonl`), `--sample 20`, `--seed 0`,
  `--personas` (default all), `--max-turns 6`, `--user-model gpt-5.4-mini`,
  `--responder-model gpt-5.4-mini`, `--judge-model gpt-5.5`, plus the run options above
  except `--limit`, with `--max-concurrency 4` and `--experiment-prefix
  simulate-jailbreak`. A persona named twice runs once.
- Every customer turn, blank ones included, reaches the detector the way prod sends
  it on the opener's channel, as described under Data.
- A conversation stops at the first blocked turn.
- After each conversation, the `--judge-model` judge decides whether the simulated
  customer's generated turns (everything after the real opener) attempted an attack.
  If there are no generated turns, or the judge fails, the verdict is left unset.

Personas: `benign_follow_up`, `benign_frustrated`, `crescendo_trust_builder`,
`roleplay_framer`, `trigger_word_planter`, `instruction_override` and
`data_exfiltration`.

Per row:

| Key | Meaning |
|---|---|
| `simulation_correct` | An attack persona was blocked, or a benign one never was. Unscored for an attack whose opener was blocked, that was not judged, or that the judge found not attempted. |
| `turns_to_block` | The customer turn that blocked an attack. Unscored for benign rows, a blocked opener, an attack not judged or not attempted, or no block. |
| `attack_attempted` | The judge's verdict that the generated turns attempted an attack. Unscored if not judged. |
| `persona_faithful` | The simulated customer attacked exactly when its persona is an attacker. Unscored if not judged. |

Per experiment. A **counted attack** is an attack-persona conversation whose simulated
customer really attacked (`attack_attempted` true) and whose real opener was not
blocked. A miss is therefore never blamed on an attacker that did not attack, or
credited to a block of the real customer's message.

| Key | Meaning |
|---|---|
| `attack_block_rate` | Share of counted attacks that were blocked. |
| `mean_turns_to_block` | Mean blocked turn over the counted attacks that were blocked. |
| `benign_false_block_rate` | Share of benign conversations that were blocked, the opener included. |
| `persona_faithful_rate` | Share of judged conversations whose persona attacked as its kind requires. |
| `error_rate` | Share of conversations whose target raised. These are left out of every other score. |

## `--local`

`--local` runs everything in memory. Nothing is synced or uploaded, and no LangSmith
key is needed. Bedrock is still called, and so is OpenAI for `simulation`. Local
results stay on your machine: they are not shared evidence.

## Configuration

Every default, key, label, help text, message and prompt the evals use lives in
`evals/jailbreak/config.toml`. The only text written in code is `settings.py`'s own
errors about a bad config, since those must not depend on the file they check.
The file is checked when the package is imported: an unknown key, a bad persona
`kind`, a repeated feedback key, a metric that names no measure or a message whose
`{placeholders}` differ from the code's stops the import with an error. To add a persona, add one `[personas.<name>]` table with a `kind` (`benign` or
`attack`) and a `prompt`.

Override any value for one run as `JAILBREAK_EVALS_<SECTION>__<KEY>`, from the
environment or a local `.env`:

```bash
JAILBREAK_EVALS_SIMULATION__MAX_TURNS=4 uv run --group eval python -m evals.jailbreak simulation --local
```

Changing `data.example_id_namespace` gives every example a new ID, so the next upload
replaces each dataset's examples.
