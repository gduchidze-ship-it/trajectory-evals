import type { ErrorInfo, JobState, LogLine, RowDone, RunJob, RunResult } from "./types";
import type { StreamEvent } from "./sse";

export interface LiveRun {
  state: JobState;
  experiment: string | null;
  experiments: number;
  done: number;
  total: number | null;
  rows: RowDone[];
  logs: LogLine[];
  result: RunResult | null;
  error: ErrorInfo | null;
}

export const MAX_LOGS = 500;

export function initialRun(job?: RunJob): LiveRun {
  return {
    state: job?.state ?? "queued",
    experiment: job?.progress.experiment ?? null,
    experiments: job?.progress.experiments ?? 0,
    done: job?.progress.done ?? 0,
    total: job?.progress.total ?? null,
    rows: [],
    logs: [],
    result: job?.result ?? null,
    error: job?.error ?? null,
  };
}

export function reduceRun(s: LiveRun, e: StreamEvent): LiveRun {
  switch (e.event) {
    case "state":
      return { ...s, state: e.data.state };
    case "experiment":
      return { ...s, experiment: e.data.experiment, total: e.data.total, done: 0, experiments: e.data.index };
    case "row":
      return { ...s, experiment: e.data.experiment, done: e.data.done, total: e.data.total, rows: [...s.rows, e.data] };
    case "log":
      return { ...s, logs: [...s.logs, e.data].slice(-MAX_LOGS) };
    case "completed":
      return { ...s, state: "completed", result: e.data.result };
    case "failed":
      return { ...s, state: "failed", error: e.data.error };
    case "cancelled":
      return { ...s, state: "cancelled" };
  }
}

/** Running pass rate over passed and failed rows only. */
export function runningPassRate(rows: readonly RowDone[]): number | null {
  const scored = rows.filter((r) => r.status === "passed" || r.status === "failed");
  return scored.length ? scored.filter((r) => r.status === "passed").length / scored.length : null;
}
