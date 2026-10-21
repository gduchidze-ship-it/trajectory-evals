import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useReducer, useState } from "react";
import { ApiError, api } from "./client";
import { initialRun, reduceRun, type LiveRun } from "./runState";
import { followRun } from "./sse";
import { TERMINAL_STATES, type RunJob } from "./types";

export type Connection = "connecting" | "live" | "reconnecting" | "closed";

export type RunStreamState =
  | { kind: "loading" }
  | { kind: "not-found" }
  | { kind: "error"; error: ApiError | Error }
  | { kind: "ready"; job: RunJob; run: LiveRun; connection: Connection };

/** Loads the job, then follows its events (replayed from id 1) until a terminal event. */
export function useRunStream(jobId: string): RunStreamState {
  const qc = useQueryClient();
  const [job, setJob] = useState<RunJob | null>(null);
  const [fatal, setFatal] = useState<"not-found" | Error | null>(null);
  const [connection, setConnection] = useState<Connection>("connecting");
  const [run, dispatch] = useReducer(reduceRun, undefined, () => initialRun());

  useEffect(() => {
    const ctl = new AbortController();
    setJob(null);
    setFatal(null);
    setConnection("connecting");
    void (async () => {
      try {
        const j = await api.run(jobId, ctl.signal);
        if (ctl.signal.aborted) return;
        setJob(j);
        const outcome = await followRun({
          jobId,
          signal: ctl.signal,
          onEvent: (e) => dispatch(e),
          onConnection: setConnection,
        });
        if (outcome !== "aborted") setConnection("closed");
        // A 204 means every event was already delivered: re-read the job so its state and result link are final.
        if (outcome === "no-content") setJob(await api.run(jobId, ctl.signal));
        void qc.invalidateQueries({ queryKey: ["runs"] });
        void qc.invalidateQueries({ queryKey: ["results"] });
        void qc.invalidateQueries({ queryKey: ["datasets"] });
      } catch (e) {
        if (ctl.signal.aborted) return;
        if (e instanceof ApiError && e.status === 404) setFatal("not-found");
        else setFatal(e instanceof Error ? e : new Error("Unknown error"));
      }
    })();
    return () => ctl.abort();
  }, [jobId, qc]);

  if (fatal === "not-found") return { kind: "not-found" };
  if (fatal) return { kind: "error", error: fatal };
  if (!job) return { kind: "loading" };
  const merged: LiveRun =
    TERMINAL_STATES.includes(job.state) && !TERMINAL_STATES.includes(run.state)
      ? { ...initialRun(job), rows: run.rows, counts: run.counts, logs: run.logs }
      : run;
  return { kind: "ready", job, run: merged, connection };
}
