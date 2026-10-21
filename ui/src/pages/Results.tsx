import { useSearchParams, Link } from "react-router-dom";
import { useAgents, useResults } from "../api/hooks";
import type { ResultKind } from "../api/types";
import { ResultsTable } from "../components/ResultsTable";
import { EmptyState, ErrorState, Page, SkeletonRows } from "../components/ui";
import { launchHref } from "../lib/links";

const KINDS: ResultKind[] = ["eval", "simulate", "redteam", "optimize", "sweep"];
const LIMIT = 50;

export function ResultsPage() {
  const [sp, setSp] = useSearchParams();
  const agent = sp.get("agent") ?? "";
  const kindParam = sp.get("kind");
  const kind = KINDS.find((k) => k === kindParam) ?? "";
  const page = Math.max(0, Number(sp.get("page") ?? 0) || 0);
  const agents = useAgents();
  const q = useResults({ agent: agent || undefined, kind, offset: page * LIMIT, limit: LIMIT });
  const set = (k: string, v: string) => {
    const n = new URLSearchParams(sp);
    if (v) n.set(k, v);
    else n.delete(k);
    if (k !== "page") n.delete("page");
    setSp(n, { replace: true });
  };
  const names = [...new Set(agents.data?.map((a) => a.name) ?? [])];
  const filtered = agent !== "" || kind !== "";

  return (
    <Page
      title="Results"
      subtitle="Every finished run in results/. Open one for its rows and transcripts, or compare two."
      actions={<Link className="btn" to="/compare">Compare runs</Link>}
    >
      <div className="mb-4 flex flex-wrap items-end gap-4">
        <div className="grid gap-1">
          <label htmlFor="agent" className="text-[13px] font-medium">Agent</label>
          <select id="agent" className="input min-w-[200px]" value={agent} onChange={(e) => set("agent", e.target.value)}>
            <option value="">All agents</option>
            {names.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="grid gap-1">
          <label htmlFor="kind" className="text-[13px] font-medium">Kind</label>
          <select id="kind" className="input min-w-[160px]" value={kind} onChange={(e) => set("kind", e.target.value)}>
            <option value="">All kinds</option>
            {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </div>
      </div>
      {q.isPending ? (
        <SkeletonRows rows={6} label="Loading results" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} title="Could not list results" />
      ) : q.data.total === 0 ? (
        filtered ? (
          <EmptyState title="No runs match these filters" action={<button className="btn" onClick={() => setSp({}, { replace: true })}>Clear filters</button>} />
        ) : (
          <EmptyState title="No results yet" action={<Link className="btn btn-primary" to={launchHref({})}>Launch a run</Link>}>
            Finished eval, simulate, red-team, optimize and sweep runs appear here.
          </EmptyState>
        )
      ) : (
        <>
          {q.data.items.length ? <ResultsTable items={q.data.items} /> : <EmptyState title="No runs on this page">Go back to an earlier page.</EmptyState>}
          {q.data.total > LIMIT ? (
            <div className="mt-3 flex items-center justify-end gap-2 text-[13px]">
              <button className="btn btn-sm" disabled={page === 0} onClick={() => set("page", String(page - 1))}>Previous</button>
              <span aria-live="polite">{page * LIMIT + 1} to {Math.min(q.data.total, (page + 1) * LIMIT)} of {q.data.total}</span>
              <button className="btn btn-sm" disabled={(page + 1) * LIMIT >= q.data.total} onClick={() => set("page", String(page + 1))}>Next</button>
            </div>
          ) : null}
        </>
      )}
    </Page>
  );
}
