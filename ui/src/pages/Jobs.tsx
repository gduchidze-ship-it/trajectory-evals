import { Link } from "react-router-dom";
import { useRuns } from "../api/hooks";
import { Chip, EmptyState, ErrorState, Page, SkeletonRows, Spinner, kindLabel } from "../components/ui";
import { relTime } from "../lib/format";
import { jobHref, launchHref } from "../lib/links";

export function JobsPage() {
  const q = useRuns();
  return (
    <Page title="Jobs" subtitle="Runs started from this server. They live in memory: a restart forgets them, finished experiments stay in Results.">
      {q.isPending ? (
        <SkeletonRows rows={3} label="Loading jobs" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : q.data.length === 0 ? (
        <EmptyState title="No jobs yet" action={<Link className="btn btn-primary" to={launchHref({})}>Launch a run</Link>}>
          Runs you start show up here while they work.
        </EmptyState>
      ) : (
        <div className="card overflow-x-auto">
          <table className="tbl">
            <caption className="sr-only">Jobs, newest first</caption>
            <thead><tr><th>Job</th><th>Command</th><th>Agent</th><th>State</th><th className="num">Progress</th><th>Started</th></tr></thead>
            <tbody>
              {q.data.map((j) => (
                <tr key={j.id} className="row-link">
                  <td><Link to={jobHref(j.id)} className="mono text-[13px] font-medium">{j.id}</Link></td>
                  <td>{kindLabel[j.command] ?? j.command}</td>
                  <td>{j.agent}</td>
                  <td>
                    <Chip tone={j.state === "running" ? "accent" : j.state === "completed" ? "pass" : j.state === "failed" ? "fail" : j.state === "cancelled" ? "warn" : undefined}>
                      {j.state === "running" ? <Spinner size={12} /> : null}{j.state}
                    </Chip>
                  </td>
                  <td className="num">{j.progress.total !== null ? `${j.progress.done} / ${j.progress.total}` : "–"}</td>
                  <td>{relTime(j.started_at ?? j.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Page>
  );
}
