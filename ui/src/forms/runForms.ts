import type { Capabilities, Command, DatasetKind, RunRequest } from "../api/types";

export type FieldKind = "agent" | "data" | "include" | "personas" | "int" | "float" | "text" | "bool";
export type FormValue = string | boolean | string[];
export type FormValues = Record<string, FormValue>;

export interface FieldSpec {
  name: string;
  label: string;
  kind: FieldKind;
  help?: string;
  required?: boolean;
  dataKind?: DatasetKind;
  advanced?: boolean;
  /** Upload toggles: disabled and forced off without a LangSmith key. */
  upload?: boolean;
}

export interface CommandSpec {
  command: Command;
  label: string;
  blurb: string;
  capability: keyof Capabilities | null;
  langsmith: boolean;
  fields: FieldSpec[];
}

const experiment: FieldSpec[] = [
  { name: "local", label: "Keep results local", kind: "bool", help: "Skip LangSmith upload; results stay in results/.", upload: true },
  { name: "limit", label: "Limit", kind: "int", help: "Run only the first N cases.", advanced: true },
  { name: "repetitions", label: "Repetitions", kind: "int", help: "Runs per case.", advanced: true },
  { name: "concurrency", label: "Concurrency", kind: "int", help: "Cases in flight at once.", advanced: true },
  { name: "prefix", label: "Experiment prefix", kind: "text", advanced: true },
];

const lsFields: FieldSpec[] = [
  { name: "project", label: "LangSmith project", kind: "text", help: "Defaults to [langsmith].project in agent.toml." },
  { name: "since_days", label: "Since (days)", kind: "int", help: "Defaults to [langsmith].since_days." },
  { name: "limit", label: "Run limit", kind: "int" },
  { name: "filter", label: "Filter", kind: "text", help: "A LangSmith filter expression.", advanced: true },
];

export const COMMAND_SPECS: Record<Command, CommandSpec> = {
  eval: {
    command: "eval",
    label: "Eval",
    blurb: "Score the agent against a golden dataset.",
    capability: null,
    langsmith: false,
    fields: [
      { name: "agent", label: "Agent", kind: "agent", required: true },
      { name: "data", label: "Golden dataset", kind: "data", dataKind: "golden", required: true },
      ...experiment,
      { name: "criteria_model", label: "Criteria model", kind: "text", help: "Overrides the judge model for criteria.", advanced: true },
    ],
  },
  simulate: {
    command: "simulate",
    label: "Simulate",
    blurb: "Drive multi-turn conversations from scenarios with simulated users.",
    capability: "simulate",
    langsmith: false,
    fields: [
      { name: "agent", label: "Agent", kind: "agent", required: true },
      { name: "data", label: "Scenarios file", kind: "data", dataKind: "scenarios", required: true },
      ...experiment,
      { name: "max_turns", label: "Max turns", kind: "int", advanced: true },
      { name: "criteria_model", label: "Criteria model", kind: "text", advanced: true },
    ],
  },
  redteam: {
    command: "redteam",
    label: "Red-team",
    blurb: "Attack the agent with adversarial personas and measure how often they succeed.",
    capability: "redteam",
    langsmith: false,
    fields: [
      { name: "agent", label: "Agent", kind: "agent", required: true },
      { name: "personas", label: "Attack personas", kind: "personas", help: "None selected runs the configured set." },
      ...experiment,
      { name: "attempts", label: "Attempts per persona", kind: "int", advanced: true },
      { name: "max_turns", label: "Max turns", kind: "int", advanced: true },
    ],
  },
  optimize: {
    command: "optimize",
    label: "Optimize",
    blurb: "Reflective hill-climb over the agent's prompts, judged on val and holdout splits.",
    capability: "optimize",
    langsmith: false,
    fields: [
      { name: "agent", label: "Agent", kind: "agent", required: true },
      { name: "data", label: "Golden dataset", kind: "data", dataKind: "golden", required: true },
      { name: "local", label: "Keep results local", kind: "bool", upload: true },
      { name: "upload_as_experiments", label: "Upload candidates as LangSmith experiments", kind: "bool", upload: true },
    ],
  },
  sweep: {
    command: "sweep",
    label: "Sweep",
    blurb: "Pick the cheapest declared prompt variant that clears the floors.",
    capability: "optimize",
    langsmith: false,
    fields: [
      { name: "agent", label: "Agent", kind: "agent", required: true },
      { name: "data", label: "Golden dataset", kind: "data", dataKind: "golden", required: true },
      { name: "include", label: "Also include prompts from", kind: "include", help: "A finished optimize run's best/ folder." },
      { name: "local", label: "Keep results local", kind: "bool", upload: true },
      { name: "upload_as_experiments", label: "Upload variants as LangSmith experiments", kind: "bool", upload: true },
    ],
  },
  harvest: {
    command: "harvest",
    label: "Harvest",
    blurb: "Pull production runs from LangSmith into a dataset awaiting review.",
    capability: "langsmith",
    langsmith: true,
    fields: [
      { name: "agent", label: "Agent", kind: "agent", required: true },
      ...lsFields,
      { name: "only_errors", label: "Only errored runs", kind: "bool" },
      { name: "feedback_key", label: "Feedback key", kind: "text", advanced: true },
      { name: "feedback_below", label: "Feedback below", kind: "float", advanced: true },
      { name: "out", label: "Output path", kind: "text", help: "Defaults to data/langsmith/<agent>-<stamp>.jsonl.", advanced: true },
    ],
  },
  score: {
    command: "score",
    label: "Score",
    blurb: "Run online evaluators over LangSmith runs and write the feedback back.",
    capability: "langsmith",
    langsmith: true,
    fields: [
      { name: "agent", label: "Agent", kind: "agent", required: true },
      ...lsFields,
      { name: "dry_run", label: "Dry run (write no feedback)", kind: "bool" },
    ],
  },
};

export function defaultValues(command: Command): FormValues {
  const v: FormValues = {};
  for (const f of COMMAND_SPECS[command].fields) {
    if (f.kind === "bool") v[f.name] = f.name === "local" || f.name === "dry_run";
    else if (f.kind === "personas") v[f.name] = [];
    else v[f.name] = "";
  }
  if ("repetitions" in v) v["repetitions"] = "1";
  if ("concurrency" in v) v["concurrency"] = "4";
  if (command === "harvest" || command === "score") v["limit"] = "100";
  return v;
}

export interface FormContext {
  /** `[langsmith].project` is set on the chosen agent. */
  projectConfigured: boolean;
}

const POSITIVE_INT = /^[1-9]\d*$/;
const asString = (v: FormValue | undefined): string => (typeof v === "string" ? v.trim() : "");

export function validateForm(command: Command, values: FormValues, ctx: FormContext = { projectConfigured: true }): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const f of COMMAND_SPECS[command].fields) {
    const s = asString(values[f.name]);
    if (f.required && s === "") errors[f.name] = `${f.label} is required.`;
    else if (f.kind === "int" && s !== "" && !POSITIVE_INT.test(s)) errors[f.name] = `${f.label} must be a whole number of 1 or more.`;
    else if (f.kind === "float" && s !== "" && !Number.isFinite(Number(s))) errors[f.name] = `${f.label} must be a number.`;
  }
  if ((command === "harvest" || command === "score") && asString(values["project"]) === "" && !ctx.projectConfigured) {
    errors["project"] = "This agent has no [langsmith].project, so a project is required.";
  }
  if ((command === "optimize" || command === "sweep") && values["local"] === true && values["upload_as_experiments"] === true) {
    errors["upload_as_experiments"] = "Choose either local results or uploading as experiments, not both.";
  }
  return errors;
}

const orNull = (s: string): string | null => (s === "" ? null : s);
const intOrNull = (s: string): number | null => (s === "" ? null : Number(s));

/** Request body for POST /api/runs. Assumes validateForm returned no errors. */
export function buildRequest(command: Command, v: FormValues): RunRequest {
  const str = (k: string) => asString(v[k]);
  const bool = (k: string) => v[k] === true;
  const exp = () => ({
    local: bool("local"),
    limit: intOrNull(str("limit")),
    repetitions: Number(str("repetitions") || "1"),
    concurrency: Number(str("concurrency") || "4"),
    prefix: orNull(str("prefix")),
  });
  switch (command) {
    case "eval":
      return { command, agent: str("agent"), data: str("data"), ...exp(), criteria_model: orNull(str("criteria_model")) };
    case "simulate":
      return { command, agent: str("agent"), data: str("data"), ...exp(), max_turns: intOrNull(str("max_turns")), criteria_model: orNull(str("criteria_model")) };
    case "redteam": {
      const personas = Array.isArray(v["personas"]) ? v["personas"] : [];
      return { command, agent: str("agent"), personas: personas.length ? personas : null, ...exp(), attempts: intOrNull(str("attempts")), max_turns: intOrNull(str("max_turns")) };
    }
    case "optimize":
      return { command, agent: str("agent"), data: str("data"), local: bool("local"), upload_as_experiments: bool("upload_as_experiments") };
    case "sweep":
      return { command, agent: str("agent"), data: str("data"), local: bool("local"), upload_as_experiments: bool("upload_as_experiments"), include: orNull(str("include")) };
    case "harvest":
      return {
        command,
        agent: str("agent"),
        project: orNull(str("project")),
        since_days: intOrNull(str("since_days")),
        limit: Number(str("limit") || "100"),
        filter: orNull(str("filter")),
        only_errors: bool("only_errors"),
        feedback_key: orNull(str("feedback_key")),
        feedback_below: str("feedback_below") === "" ? null : Number(str("feedback_below")),
        out: orNull(str("out")),
      };
    case "score":
      return {
        command,
        agent: str("agent"),
        project: orNull(str("project")),
        since_days: intOrNull(str("since_days")),
        limit: Number(str("limit") || "100"),
        filter: orNull(str("filter")),
        dry_run: bool("dry_run"),
      };
  }
}
