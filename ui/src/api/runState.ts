import type { ErrorInfo, JobState, LogLine, RowDone, RowStatus, RunJob, RunResult } from "./types";
import type { StreamEvent } from "./sse";

export interface LiveRun {
  state: JobState;
  experiment: string | null;
  experiments: number;
  done: number;
  total: number | null;
  /** The most recent MAX_ROWS rows; `counts` covers every row ever seen. */
  rows: RowDone[];
  counts: Record<RowStatus, number>;
  logs: LogLine[];
  result: RunResult | null;
  error: ErrorInfo | null;
}

export const MAX_LOGS = 500;
export const MAX_ROWS = 500;
export const emptyCounts = (): Record<RowStatus, number> => ({ passed: 0, failed: 0, errored: 0, unscored: 0 });

export function initialRun(job?: RunJob): LiveRun {
  return {
    state: job?.state ?? "queued",
    experiment: job?.progress.experiment ?? null,
    experiments: job?.progress.experiments ?? 0,
    done: job?.progress.done ?? 0,
    total: job?.progress.total ?? null,
    rows: [],
    counts: emptyCounts(),
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
      return {
        ...s,
        experiment: e.data.experiment,
        done: e.data.done,
        total: e.data.total,
        rows: [...s.rows, e.data].slice(-MAX_ROWS),
        counts: { ...s.counts, [e.data.status]: s.counts[e.data.status] + 1 },
      };
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
export function runningPassRate(counts: Record<RowStatus, number>): number | null {
  const scored = counts.passed + counts.failed;
  return scored ? counts.passed / scored : null;
}
