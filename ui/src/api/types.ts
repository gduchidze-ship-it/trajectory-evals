// Hand-written from service-contract.md §A/§B. After `npm run gen:api`, replace this file's
// bodies with re-exports of components["schemas"][...] from ./schema.d.ts; nothing else imports
// schema.d.ts, so the swap stays a one-file change.

export type ErrorKind =
  | "preflight"
  | "agent_config"
  | "case_format"
  | "validation"
  | "agent"
  | "invalid"
  | "exists"
  | "not_found"
  | "langsmith_not_found"
  | "forbidden"
  | "conflict"
  | "rate_limited"
  | "langsmith"
  | "internal";

export interface ErrorInfo {
  kind: ErrorKind;
  message: string;
  input_error: boolean;
}
export interface ErrorBody {
  error: ErrorInfo;
}

export interface Page<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

export interface Health {
  status: "ok";
  version: string;
  demo: boolean;
  root: string;
  results_dir: string;
  langsmith_key: boolean;
}

// ---- agents ----
export interface AgentRef {
  id: string;
  name: string;
  path: string;
  source: "agents" | "examples";
}
export interface Capabilities {
  simulate: boolean;
  redteam: boolean;
  optimize: boolean;
  langsmith: boolean;
}
export interface AgentSummary extends AgentRef {
  description: string | null;
  endpoint: "url" | "bridge" | null;
  valid: boolean;
  error: ErrorInfo | null;
  datasets: number;
  capabilities: Capabilities;
}
export interface EndpointInfo {
  kind: "url" | "bridge";
  url: string | null;
  responder: string | null;
  history: "metadata" | "replay";
  timeout_s: number;
  auth_token_env: string | null;
  auth_token_set: boolean;
}
export interface PersonaInfo {
  name: string;
  kind: "benign" | "attack";
  source: "library" | "agent";
}
export interface JudgeInfo {
  name: string;
  prompt: string;
  model: string;
  continuous: boolean;
  reference_free: boolean;
  gate: boolean;
}
export interface AgentFile {
  path: string;
  kind: "config" | "prompt" | "other";
  editable: boolean;
}
export type DatasetKind = "golden" | "scenarios";
export interface DatasetSummary {
  path: string;
  agent_id: string | null;
  kind: DatasetKind;
  rows: number;
  invalid_rows: number;
  review_required: number;
  writable: boolean;
  modified_at: string;
}
export interface AgentDetail extends AgentSummary {
  agent: EndpointInfo | null;
  tables: Record<string, unknown>;
  judges: JudgeInfo[];
  personas: PersonaInfo[];
  simulation: Record<string, unknown> | null;
  redteam: Record<string, unknown> | null;
  optimize: Record<string, unknown> | null;
  variants: Record<string, Record<string, unknown>>;
  langsmith: Record<string, unknown> | null;
  files: AgentFile[];
  dataset_list: DatasetSummary[];
}

export type CheckStepName = "config" | "graders" | "datasets" | "agent" | "tables" | "card" | "ping";
export interface DatasetCount {
  path: string;
  cases: number;
}
export interface CheckStep {
  step: CheckStepName;
  ok: boolean;
  error: ErrorInfo | null;
  datasets: DatasetCount[];
  reply: string | null;
}
export interface CheckReport {
  agent_id: string;
  ok: boolean;
  steps: CheckStep[];
}
export interface CheckRequest {
  ping: boolean;
}
export interface FileContent {
  path: string;
  text: string;
}
export interface FileWrite {
  text: string;
}

// ---- datasets ----
export interface RowError {
  line: number;
  message: string;
}
export interface DatasetRow {
  line: number;
  id: string | null;
  data: Record<string, unknown> | null;
  error: string | null;
}
export interface DatasetPage extends Page<DatasetRow> {
  path: string;
  kind: DatasetKind;
  errors: RowError[];
}
export interface DatasetWrite {
  rows: Record<string, unknown>[];
  kind?: DatasetKind | null;
}
export interface ValidationReport {
  ok: boolean;
  rows: number;
  errors: RowError[];
}

// ---- run requests ----
export type Command = "eval" | "simulate" | "redteam" | "optimize" | "sweep" | "harvest" | "score";
export const COMMANDS: readonly Command[] = ["eval", "simulate", "redteam", "optimize", "sweep", "harvest", "score"];

export interface ExperimentOptions {
  local: boolean;
  limit: number | null;
  repetitions: number;
  concurrency: number;
  prefix: string | null;
}
export interface EvalRequest extends ExperimentOptions {
  command: "eval";
  agent: string;
  data: string;
  criteria_model: string | null;
}
export interface SimulateRequest extends ExperimentOptions {
  command: "simulate";
  agent: string;
  data: string;
  max_turns: number | null;
  criteria_model: string | null;
}
export interface RedteamRequest extends ExperimentOptions {
  command: "redteam";
  agent: string;
  personas: string[] | null;
  attempts: number | null;
  max_turns: number | null;
}
export interface OptimizeRequest {
  command: "optimize";
  agent: string;
  data: string;
  local: boolean;
  upload_as_experiments: boolean;
}
export interface SweepRequest extends Omit<OptimizeRequest, "command"> {
  command: "sweep";
  include: string | null;
}
export interface HarvestRequest {
  command: "harvest";
  agent: string;
  project: string | null;
  since_days: number | null;
  limit: number;
  filter: string | null;
  only_errors: boolean;
  feedback_key: string | null;
  feedback_below: number | null;
  out: string | null;
}
export interface ScoreRequest {
  command: "score";
  agent: string;
  project: string | null;
  since_days: number | null;
  limit: number;
  filter: string | null;
  dry_run: boolean;
}
export type RunRequest =
  | EvalRequest
  | SimulateRequest
  | RedteamRequest
  | OptimizeRequest
  | SweepRequest
  | HarvestRequest
  | ScoreRequest;

// ---- run results ----
export type RowStatus = "passed" | "failed" | "errored" | "unscored";
export type ResultKind = "eval" | "simulate" | "redteam" | "optimize" | "sweep";

export interface ExperimentResult {
  kind: "experiment";
  command: "eval" | "simulate" | "redteam";
  run_id: string;
  run_dir: string;
  experiment: string;
  dataset: string;
  url: string | null;
  summary: Record<string, number | null>;
}
export interface OptimizeResult {
  kind: "optimize" | "sweep";
  run_id: string;
  run_dir: string;
  report: OptimizeReport | SweepReport;
}
export interface HarvestResult {
  kind: "harvest";
  path: string;
  rows: number;
}
export interface KeyCounts {
  scored: number;
  skipped: number;
  errors: number;
  written: number;
}
export interface ScoreResult {
  kind: "score";
  runs: number;
  keys: Record<string, KeyCounts>;
}
export type RunResult = ExperimentResult | OptimizeResult | HarvestResult | ScoreResult;

export type JobState = "queued" | "running" | "completed" | "failed" | "cancelled";
export const TERMINAL_STATES: readonly JobState[] = ["completed", "failed", "cancelled"];
export interface Progress {
  experiment: string | null;
  done: number;
  total: number | null;
  experiments: number;
}
export interface RunJob {
  id: string;
  command: Command;
  agent: string;
  state: JobState;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  request: RunRequest;
  progress: Progress;
  result: RunResult | null;
  error: ErrorInfo | null;
  last_event_id: number;
}

// ---- events ----
export interface StateChanged {
  state: JobState;
}
export interface ExperimentStarted {
  experiment: string;
  total: number;
  index: number;
}
export interface RowDone {
  experiment: string;
  done: number;
  total: number;
  case_id: string;
  status: RowStatus;
  latency_ms: number | null;
  feedback: Record<string, number | boolean | null>;
}
export interface LogLine {
  level: "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";
  logger: string;
  message: string;
}
export interface Completed {
  result: RunResult;
}
export interface Failed {
  error: ErrorInfo;
}
export type Cancelled = Record<string, never>;

export type EventName = "state" | "experiment" | "row" | "log" | "completed" | "failed" | "cancelled";
export type RunEventPayload = StateChanged | ExperimentStarted | RowDone | LogLine | Completed | Failed | Cancelled;
export interface RunEvent {
  id: number;
  time: string;
  event: EventName;
  data: RunEventPayload;
}

// ---- results ----
export interface ResultSummary {
  run_id: string;
  kind: ResultKind;
  agent: string | null;
  created_at: string;
  complete: boolean;
  experiment: string | null;
  dataset: string | null;
  dataset_file: string | null;
  url: string | null;
  local: boolean | null;
  rows: number | null;
  pass_rate: number | null;
  pass_rate_ci_low: number | null;
  pass_rate_ci_high: number | null;
  error_rate: number | null;
  attack_succeeded: number | null;
  attack_succeeded_coverage: number | null;
  verdict: string | null;
  best: string | null;
}
export interface FeedbackStat {
  key: string;
  mean: number | null;
  coverage: number | null;
}
export interface PersonaStat {
  persona: string;
  kind: "benign" | "attack" | null;
  rows: number;
  errors: number;
  attack_succeeded: number | null;
  coverage: number;
  turns_to_break: number | null;
  defended: number | null;
}
export interface ExperimentDetail extends ResultSummary {
  detail: "experiment";
  summary: Record<string, number | null>;
  meta: Record<string, unknown>;
  feedback: FeedbackStat[];
  segments: Record<string, Record<string, number | null>>;
  personas: PersonaStat[];
  pass_keys: string[] | null;
}
export interface CandidateRecord {
  id: string;
  parent: string | null;
  verdict: string;
  reason: string | null;
  rationale: string | null;
  cost: number | null;
  val: Record<string, number | null> | null;
  judge_coverage: Record<string, number | null>;
  redteam: Record<string, number | null> | null;
  fixed_cases: string[];
  regressed_cases: string[];
  sign_test_p: number | null;
  deltas: Record<string, number>;
  agent_calls: number;
}
export interface Holdout {
  candidate: string;
  baseline: Record<string, number | null>;
  best: Record<string, number | null>;
  deltas: Record<string, number>;
  fixed_cases: string[];
  regressed_cases: string[];
  sign_test_p: number;
}
export interface SplitSizes {
  train: number;
  val: number;
  holdout: number;
  labelled: boolean;
}
export interface ReportHeader {
  run_id: string;
  agent: string;
  upload: boolean;
  optimize: Record<string, unknown>;
  splits: SplitSizes;
  holdout: Holdout | null;
  verdict: string | null;
  reason: string | null;
  agent_calls: number;
}
export interface OptimizeReport extends ReportHeader {
  mode: "optimize";
  candidates: CandidateRecord[];
  best: string;
  stop_reason: string | null;
}
export interface SweepReport extends ReportHeader {
  mode: "sweep";
  variants: CandidateRecord[];
  winner: string | null;
}
export interface OptimizeDetail extends ResultSummary {
  detail: "optimize";
  report: OptimizeReport | SweepReport;
  evals: string[];
}
export type ResultDetail = ExperimentDetail | OptimizeDetail;

export interface RowSummary {
  index: number;
  case_id: string;
  status: RowStatus;
  error: string | null;
  latency_ms: number | null;
  feedback: Record<string, number | boolean | null>;
  metadata: Record<string, unknown>;
  persona: string | null;
  input_preview: string;
  reply_preview: string;
}
export interface RowDetail extends RowSummary {
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  reference_outputs: Record<string, unknown>;
  comments: Record<string, string | null>;
  feedback_errors: string[];
}
export interface MetricDelta {
  key: string;
  a: number | null;
  b: number | null;
  delta: number | null;
}
export interface Comparison {
  a: ResultSummary;
  b: ResultSummary;
  metrics: MetricDelta[];
  fixed_cases: string[] | null;
  regressed_cases: string[] | null;
  sign_test_p: number | null;
  only_in_a: string[];
  only_in_b: string[];
}

// ---- langsmith ----
export interface PullRequest {
  agent: string;
  dataset: string;
  out: string;
}
export interface PushRequest {
  agent: string;
  data: string;
  dataset?: string | null;
}
export interface PullResult {
  path: string;
}
export interface PushResult {
  path: string;
  name: string;
}
