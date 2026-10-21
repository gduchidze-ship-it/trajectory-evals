import { CircleCheck, CircleX, Ban, Square, WifiOff } from "lucide-react";
import { useEffect, useRef } from "react";
import { Link, useParams } from "react-router-dom";
import { useCancelRun } from "../api/hooks";
import { runningPassRate, type LiveRun } from "../api/runState";
import { useRunStream, type Connection } from "../api/useRunStream";
import { TERMINAL_STATES, type JobState, type RunJob, type RunResult } from "../api/types";
import { Banner, Chip, EmptyState, ErrorState, Page, Progress, Section, SkeletonRows, Spinner, StatusChip, kindLabel } from "../components/ui";
import { ms, pct } from "../lib/format";
import { datasetHref, resultHref } from "../lib/links";

const stateTone: Record<JobState, "accent" | "pass" | "fail" | "warn" | undefined> = {
  queued: undefined,
  running: "accent",
  completed: "pass",
  failed: "fail",
  cancelled: "warn",
};

export function RunLivePage() {
  const { jobId = "" } = useParams();
  const s = useRunStream(jobId);
  if (s.kind === "loading") return <Page title="Run"><SkeletonRows rows={5} label="Loading run" /></Page>;
  if (s.kind === "not-found") {
    return (
      <Page title="Run not found" back={<Link to="/runs">Jobs</Link>}>
        <EmptyState title="The server does not know this job">
          Jobs live in memory, so a server restart forgets them. Finished experiments are still on disk: <Link to="/results">open Results</Link>.
        </EmptyState>
      </Page>
    );
  }
  if (s.kind === "error") return <Page title="Run"><ErrorState error={s.error} title="Could not follow this run" /></Page>;
  return <RunView job={s.job} run={s.run} connection={s.connection} />;
}

function RunView({ job, run, connection }: { job: RunJob; run: LiveRun; connection: Connection }) {
  const cancel = useCancelRun();
  const terminal = TERMINAL_STATES.includes(run.state);
  const multi = job.command === "optimize" || job.command === "sweep";
  const rate = runningPassRate(run.rows);
  const logRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    const el = logRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [run.logs.length]);

  return (
    <Page
      title={`${kindLabel[job.command] ?? job.command} · ${job.agent.split(".").pop() ?? job.agent}`}
      back={<Link to="/runs">Jobs</Link>}
      subtitle={<span className="mono text-[12.5px]">job {job.id}</span>}
      actions={
        <>
          <Chip tone={stateTone[run.state]}>
            {run.state === "running" ? <Spinner size={12} /> : null}
            {run.state}
          </Chip>
          {connection === "reconnecting" && !terminal ? <Chip tone="warn"><WifiOff size={12} aria-hidden="true" /> reconnecting</Chip> : null}
          <button type="button" className="btn btn-danger" disabled={terminal || cancel.isPending} onClick={() => cancel.mutate(job.id)}>
            <Square size={13} aria-hidden="true" /> {cancel.isPending ? "Cancelling" : "Cancel"}
          </button>
        </>
      }
    >
      <div role="status" className="sr-only">{`Job ${run.state}`}</div>
      {cancel.isError ? <Banner tone="error">{cancel.error.message}</Banner> : null}
      <Terminal job={job} run={run} />

      <Section title="Progress">
        <div className="card card-pad grid gap-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2 text-[14px]">
            <span className="mono text-[13px]">{run.experiment ?? "waiting for the first experiment"}</span>
            <span className="tabular-nums">{run.done}{run.total !== null ? ` / ${run.total}` : ""} rows</span>
          </div>
          <Progress value={run.done} max={run.total} label="Rows finished in the current experiment" />
          <dl className="m-0 flex flex-wrap gap-x-8 gap-y-1 text-[14px]">
            {multi ? (<div><dt className="muted inline">Experiments started </dt><dd className="m-0 inline font-medium tabular-nums">{run.experiments}</dd></div>) : null}
            {!multi && (job.command === "eval" || job.command === "simulate" || job.command === "redteam") ? (
              <div><dt className="muted inline">Running pass rate (passed and failed rows) </dt><dd className="m-0 inline font-medium tabular-nums">{pct(rate)}</dd></div>
            ) : null}
          </dl>
        </div>
      </Section>

      <div className="grid items-start gap-5 xl:grid-cols-2">
        <Section title={`Rows (${run.rows.length})`}>
          {run.rows.length === 0 ? (
            <div className="card card-pad muted text-[14px]">{terminal ? "This job reported no rows." : "Rows appear here as they finish."}</div>
          ) : (
            <div className="card max-h-[420px] overflow-auto">
              <table className="tbl">
                <caption className="sr-only">Finished rows, newest last</caption>
                <thead><tr><th>Case</th><th>Status</th><th className="num">Latency</th></tr></thead>
                <tbody>
                  {run.rows.map((r, i) => (
                    <tr key={i}>
                      <td className="mono text-[13px]">{r.case_id}</td>
                      <td><StatusChip status={r.status} /></td>
                      <td className="num">{ms(r.latency_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Section>
        <Section title="Log">
          <div
            ref={logRef}
            onScroll={(e) => { const t = e.currentTarget; stick.current = t.scrollHeight - t.scrollTop - t.clientHeight < 24; }}
            className="card mono max-h-[420px] min-h-[96px] overflow-auto p-3 text-[12.5px]"
            role="log"
            aria-label="Job log"
            tabIndex={0}
          >
            {run.logs.length === 0 ? <span className="muted">No log lines yet.</span> : run.logs.map((l, i) => (
              <div key={i} className="pre-text">
                <span className={l.level === "ERROR" || l.level === "CRITICAL" ? "text-fail" : l.level === "WARNING" ? "text-warn" : "muted"}>{l.level.padEnd(7)}</span> {l.message}
              </div>
            ))}
          </div>
        </Section>
      </div>
    </Page>
  );
}

function ResultLinks({ result }: { result: RunResult }) {
  switch (result.kind) {
    case "experiment":
      return <Link className="btn btn-primary" to={resultHref(result.run_id)}>Open results</Link>;
    case "optimize":
    case "sweep":
      return <Link className="btn btn-primary" to={resultHref(result.run_id)}>Open the {result.kind} report</Link>;
    case "harvest":
      return (
        <div className="flex flex-wrap items-center gap-3">
          <span>Wrote {result.rows} rows to <code>{result.path}</code>.</span>
          <Link className="btn btn-primary" to={datasetHref(result.path)}>Review them</Link>
          <Link className="btn" to="/langsmith?tab=review">Open the review queue</Link>
        </div>
      );
    case "score":
      return (
        <div className="w-full">
          <p className="m-0 mb-2">Scored {result.runs} runs.</p>
          <div className="card overflow-x-auto bg-surface">
            <table className="tbl">
              <caption className="sr-only">Counts per feedback key</caption>
              <thead><tr><th>Key</th><th className="num">Scored</th><th className="num">Skipped</th><th className="num">Errors</th><th className="num">Written</th></tr></thead>
              <tbody>
                {Object.entries(result.keys).map(([k, c]) => (
                  <tr key={k}><td className="mono text-[13px]">{k}</td><td className="num">{c.scored}</td><td className="num">{c.skipped}</td><td className="num">{c.errors}</td><td className="num">{c.written}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
  }
}

function Terminal({ job, run }: { job: RunJob; run: LiveRun }) {
  if (run.state === "completed" && run.result) {
    return (
      <Banner tone="success" role="status">
        <div className="flex flex-wrap items-center gap-3">
          <span className="inline-flex items-center gap-1.5 font-semibold"><CircleCheck size={16} aria-hidden="true" /> Completed.</span>
          <ResultLinks result={run.result} />
        </div>
      </Banner>
    );
  }
  if (run.state === "failed") {
    return (
      <Banner tone="error">
        <div className="inline-flex items-center gap-1.5 font-semibold"><CircleX size={16} aria-hidden="true" /> Failed</div>
        <p className="pre-text m-0 mt-1">{run.error?.message ?? "The job failed."}</p>
        {run.error?.input_error ? <p className="m-0 mt-1">This is a problem with the input. Fix it and <Link to={`/launch?command=${job.command}&agent=${encodeURIComponent(job.agent)}`}>launch again</Link>.</p> : null}
      </Banner>
    );
  }
  if (run.state === "cancelled") {
    return (
      <Banner tone="warn">
        <div className="inline-flex items-center gap-1.5 font-semibold"><Ban size={16} aria-hidden="true" /> Cancelled</div>
        <p className="m-0 mt-1">
          {job.command === "optimize" || job.command === "sweep"
            ? "Optimize and sweep keep a partial report on disk. It shows as incomplete in Results."
            : "Nothing was written for this job."}
        </p>
      </Banner>
    );
  }
  return null;
}
