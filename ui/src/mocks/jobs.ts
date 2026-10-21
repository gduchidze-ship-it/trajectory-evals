import type { Command, ErrorInfo, EventName, RowDone, RunEvent, RunEventPayload, RunJob, RunRequest, RunResult } from "../api/types";
import { E2_ID, O1_ID, R1_ID, S1_ID, SW1_ID } from "./fixtures";
import { RESULTS } from "./fixtures";

export const mockConfig = { rowDelayMs: 220 };

export interface MockJob {
  job: RunJob;
  events: RunEvent[];
  listeners: Set<() => void>;
  timer: ReturnType<typeof setTimeout> | null;
}

export const jobs = new Map<string, MockJob>();
export const FIXTURE_JOB_ID = "7a1c3e5f9b02";
const TERMINAL = new Set(["completed", "failed", "cancelled"]);
let counter = 0;

const nowIso = () => new Date().toISOString();

export function push(m: MockJob, event: EventName, data: RunEventPayload): void {
  const id = m.events.length + 1;
  m.events.push({ id, time: nowIso(), event, data });
  m.job.last_event_id = id;
  for (const l of [...m.listeners]) l();
}

export const isTerminalJob = (m: MockJob): boolean => TERMINAL.has(m.job.state);

function resultFor(req: RunRequest): RunResult {
  switch (req.command) {
    case "eval":
    case "simulate":
    case "redteam": {
      const id = req.command === "eval" ? E2_ID : req.command === "simulate" ? S1_ID : R1_ID;
      const r = RESULTS.find((x) => x.run_id === id);
      return {
        kind: "experiment",
        command: req.command,
        run_id: id,
        run_dir: `results/demo/${id}`,
        experiment: r?.experiment ?? "experiment",
        dataset: r?.dataset ?? "dataset",
        url: null,
        summary: { pass_rate: r?.pass_rate ?? null, rows: r?.rows ?? null },
      };
    }
    case "optimize":
    case "sweep": {
      const id = req.command === "optimize" ? O1_ID : SW1_ID;
      const r = RESULTS.find((x) => x.run_id === id);
      if (r?.detail !== "optimize") throw new Error("fixture missing");
      return { kind: req.command, run_id: id, run_dir: `results/demo/${id}`, report: r.report };
    }
    case "harvest":
      return { kind: "harvest", path: "data/langsmith/sales_assistant-20261021_120000.jsonl", rows: 6 };
    case "score":
      return {
        kind: "score",
        runs: 30,
        keys: {
          helpfulness: { scored: 21, skipped: 5, errors: 4, written: req.dry_run ? 0 : 21 },
          tone: { scored: 24, skipped: 6, errors: 0, written: req.dry_run ? 0 : 24 },
        },
      };
  }
}

interface Step {
  experiment: string;
  total: number;
}

function planFor(req: RunRequest, slow: boolean): Step[] {
  switch (req.command) {
    case "eval":
    case "simulate":
    case "redteam": {
      const total = slow ? 12 : Math.min(("limit" in req && req.limit) || 5, 6);
      return [{ experiment: `support_agent-${req.command}`, total }];
    }
    case "optimize":
    case "sweep":
      return ["baseline-val", "candidate-1-val", "candidate-1-holdout"].map((n) => ({ experiment: `support_agent-${req.command}-${n}`, total: 3 }));
    default:
      return [];
  }
}

function step(m: MockJob, plan: Step[], si: number, ri: number, delay: number): void {
  if (isTerminalJob(m)) return;
  const next = (a: number, b: number) => {
    m.timer = setTimeout(() => step(m, plan, a, b, delay), delay);
  };
  const cur = plan[si];
  if (!cur) {
    const req = m.job.request;
    if (req.command === "score") push(m, "log", { level: "INFO", logger: "trajectory_evals.langsmith_io.score", message: "scored 30 runs" });
    const result = resultFor(req);
    m.job.state = "completed";
    m.job.result = result;
    m.job.finished_at = nowIso();
    push(m, "completed", { result });
    return;
  }
  if (ri === 0) {
    m.job.progress = { experiment: cur.experiment, done: 0, total: cur.total, experiments: si + 1 };
    push(m, "experiment", { experiment: cur.experiment, total: cur.total, index: si + 1 });
    push(m, "log", { level: "INFO", logger: "trajectory_evals.core.runner", message: `started ${cur.experiment} with ${cur.total} rows` });
  }
  if (ri >= cur.total) return next(si + 1, 0);
  const status: RowDone["status"] = (ri + si) % 4 === 3 ? "failed" : ri === 4 && si === 0 && plan.length === 1 ? "errored" : "passed";
  m.job.progress = { ...m.job.progress, done: ri + 1 };
  push(m, "row", {
    experiment: cur.experiment,
    done: ri + 1,
    total: cur.total,
    case_id: ["refund-eligible", "late-delivery", "lie-about-refund", "address-change", "cancel-shipped", "angry-refund"][ri % 6] ?? `case-${ri}`,
    status,
    latency_ms: status === "errored" ? null : 4 + ri * 1.3,
    feedback: status === "errored" ? {} : { contains: status === "passed", helpful: true },
  });
  next(si, ri + 1);
}

function begin(m: MockJob, slow: boolean, delay: number): void {
  m.job.state = "running";
  m.job.started_at = nowIso();
  push(m, "state", { state: "running" });
  const req = m.job.request;
  if (req.command === "harvest") {
    push(m, "log", { level: "INFO", logger: "trajectory_evals.langsmith_io.harvest", message: "paging root runs from project dev-sales" });
    push(m, "log", { level: "WARNING", logger: "trajectory_evals.langsmith_io.harvest", message: "11 of 30 runs errored" });
  }
  m.timer = setTimeout(() => step(m, planFor(req, slow), 0, 0, delay), delay);
}

export function startJob(req: RunRequest, opts: { id?: string; slow?: boolean } = {}): MockJob {
  const id = opts.id ?? (++counter).toString(16).padStart(2, "0") + "3f9c0a1b2c4d".slice(2);
  const job: RunJob = {
    id,
    command: req.command as Command,
    agent: req.agent,
    state: "queued",
    created_at: nowIso(),
    started_at: null,
    finished_at: null,
    request: req,
    progress: { experiment: null, done: 0, total: null, experiments: 0 },
    result: null,
    error: null,
    last_event_id: 0,
  };
  const m: MockJob = { job, events: [], listeners: new Set(), timer: null };
  jobs.set(id, m);
  begin(m, opts.slow ?? false, opts.slow ? Math.max(mockConfig.rowDelayMs * 5, 0) : mockConfig.rowDelayMs);
  return m;
}

export function cancelJob(m: MockJob): boolean {
  if (isTerminalJob(m)) return false;
  if (m.timer) clearTimeout(m.timer);
  m.job.state = "cancelled";
  m.job.finished_at = nowIso();
  push(m, "cancelled", {});
  return true;
}

export function failJob(m: MockJob, error: ErrorInfo): void {
  m.job.state = "failed";
  m.job.error = error;
  push(m, "failed", { error });
}

/** The always-running job from the fixtures; started lazily so tests that never ask for it leave no timers. */
export function ensureFixtureJob(): void {
  if (jobs.has(FIXTURE_JOB_ID)) return;
  startJob(
    {
      command: "eval",
      agent: "examples.support_agent",
      data: "examples/support_agent/datasets/golden.sample.jsonl",
      local: true,
      limit: null,
      repetitions: 1,
      concurrency: 4,
      prefix: null,
      criteria_model: null,
    },
    { id: FIXTURE_JOB_ID, slow: true },
  );
}

export function resetJobs(): void {
  for (const m of jobs.values()) if (m.timer) clearTimeout(m.timer);
  jobs.clear();
  counter = 0;
}
