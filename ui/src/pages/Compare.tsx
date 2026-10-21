import { ArrowLeftRight } from "lucide-react";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useCompare, useResults } from "../api/hooks";
import type { Comparison, MetricDelta, ResultSummary } from "../api/types";
import { Banner, Chip, EmptyState, ErrorState, Page, Section, SkeletonRows } from "../components/ui";
import { dash, dateTime, isShare, num, pct, shortRun, signed } from "../lib/format";

const LOWER_IS_BETTER = (k: string) => k.includes("error") || k.includes("latency") || k === "attack_succeeded";

function fmt(k: string, v: number | null): string {
  if (v === null) return dash;
  if (k.includes("latency")) return `${num(v, 1)} ms`;
  return isShare(k) ? pct(v, 1) : num(v, 3);
}

const opt = (r: ResultSummary) => `${shortRun(r.run_id)} · ${dateTime(r.created_at)} · pass ${pct(r.pass_rate)}`;

export function ComparePage() {
  const [sp, setSp] = useSearchParams();
  const a = sp.get("a") ?? "";
  const b = sp.get("b") ?? "";
  const [all, setAll] = useState(false);
  const runs = useResults({ limit: 500 });
  const cmp = useCompare(a || null, b || null);
  const set = (k: string, v: string) => {
    const n = new URLSearchParams(sp);
    if (v) n.set(k, v);
    else n.delete(k);
    setSp(n, { replace: true });
  };

  const experiments = (runs.data?.items ?? []).filter((r) => r.kind === "eval" || r.kind === "simulate" || r.kind === "redteam");
  const A = experiments.find((r) => r.run_id === a);
  const bOptions = experiments.filter((r) => r.run_id !== a && (all || !A || (r.agent === A.agent && r.dataset_file === A.dataset_file)));

  return (
    <Page title="Compare runs" subtitle="Run A is the baseline: fixed means failing in A and passing in B.">
      {runs.isPending ? (
        <SkeletonRows rows={2} label="Loading runs" />
      ) : runs.isError ? (
        <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
      ) : experiments.length < 2 ? (
        <EmptyState title="Comparing needs two finished runs">
          Run an eval, simulate or red-team at least twice. <Link to="/launch">Launch a run</Link>.
        </EmptyState>
      ) : (
        <>
          <div className="mb-5 flex flex-wrap items-end gap-3">
            <div className="grid min-w-[260px] flex-1 gap-1">
              <label htmlFor="run-a" className="text-[13px] font-medium">Run A (baseline)</label>
              <select id="run-a" className="input" value={a} onChange={(e) => set("a", e.target.value)}>
                <option value="">Choose a run</option>
                {experiments.map((r) => <option key={r.run_id} value={r.run_id}>{opt(r)}</option>)}
              </select>
            </div>
            <button type="button" className="btn" disabled={!a || !b} onClick={() => setSp({ a: b, b: a }, { replace: true })} aria-label="Swap A and B">
              <ArrowLeftRight size={15} aria-hidden="true" /> Swap
            </button>
            <div className="grid min-w-[260px] flex-1 gap-1">
              <label htmlFor="run-b" className="text-[13px] font-medium">Run B</label>
              <select id="run-b" className="input" value={b} onChange={(e) => set("b", e.target.value)} disabled={!a}>
                <option value="">{a ? "Choose a run" : "Choose run A first"}</option>
                {bOptions.map((r) => <option key={r.run_id} value={r.run_id}>{opt(r)}</option>)}
              </select>
            </div>
            <label className="flex items-center gap-2 pb-2 text-[13px]">
              <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
              Offer runs of other agents or datasets
            </label>
          </div>
          {!a || !b ? (
            <EmptyState title="Pick two runs">Only runs of the same agent and dataset file are offered by default.</EmptyState>
          ) : cmp.isPending ? (
            <SkeletonRows rows={6} label="Comparing" />
          ) : cmp.isError ? (
            <ErrorState error={cmp.error} onRetry={() => void cmp.refetch()} title="Could not compare these runs" />
          ) : (
            <ComparisonView c={cmp.data} />
          )}
        </>
      )}
    </Page>
  );
}

function Delta({ m }: { m: MetricDelta }) {
  if (m.delta === null) return <span className="muted">{dash}</span>;
  const lower = LOWER_IS_BETTER(m.key);
  const good = lower ? m.delta < 0 : m.delta > 0;
  const bad = lower ? m.delta > 0 : m.delta < 0;
  const share = isShare(m.key);
  return (
    <span className={good ? "font-medium text-pass" : bad ? "font-medium text-fail" : "muted"}>
      {m.delta === 0 ? "no change" : `${good ? "better " : "worse "}${signed(m.delta, share)}`}
    </span>
  );
}

function CaseList({ title, cases, a, b, tone }: { title: string; cases: string[]; a: string; b: string; tone: "pass" | "fail" }) {
  return (
    <section className="card card-pad" aria-label={title}>
      <h3 className="mb-2 flex items-center gap-2 text-[15px]">{title} <Chip tone={cases.length ? tone : undefined}>{cases.length}</Chip></h3>
      {cases.length === 0 ? <p className="muted m-0 text-[14px]">None.</p> : (
        <ul className="m-0 grid list-none gap-1.5 p-0">
          {cases.map((c) => (
            <li key={c} className="flex flex-wrap items-center gap-3 text-[14px]">
              <span className="mono text-[13px]">{c}</span>
              <Link to={`/results/${encodeURIComponent(a)}?q=${encodeURIComponent(c)}#rows`} className="text-[13px]">row in A</Link>
              <Link to={`/results/${encodeURIComponent(b)}?q=${encodeURIComponent(c)}#rows`} className="text-[13px]">row in B</Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function ComparisonView({ c }: { c: Comparison }) {
  const noKeys = c.fixed_cases === null || c.regressed_cases === null;
  return (
    <>
      {noKeys ? <Banner tone="warn">Fixed and regressed cases need both runs recorded with pass keys. At least one of these predates that, so only the metric deltas are shown.</Banner> : null}
      <div className="mb-6 grid gap-3 md:grid-cols-2">
        {([["A (baseline)", c.a], ["B", c.b]] as const).map(([label, r]) => (
          <div key={label} className="card card-pad">
            <div className="eyebrow">Run {label}</div>
            <Link to={`/results/${encodeURIComponent(r.run_id)}`} className="font-medium [overflow-wrap:anywhere]">{shortRun(r.run_id)}</Link>
            <div className="muted text-[13px]">{r.agent} · {r.dataset_file} · {dateTime(r.created_at)} · {r.rows ?? dash} rows</div>
          </div>
        ))}
      </div>
      <Section title="Metrics" id="metrics">
        <div className="card overflow-x-auto">
          <table className="tbl">
            <caption className="sr-only">Metric changes from A to B</caption>
            <thead><tr><th>Metric</th><th className="num">A</th><th className="num">B</th><th>Change</th></tr></thead>
            <tbody>
              {c.metrics.map((m) => (
                <tr key={m.key}>
                  <td className="mono text-[13px]">{m.key}</td>
                  <td className="num">{fmt(m.key, m.a)}</td>
                  <td className="num">{fmt(m.key, m.b)}</td>
                  <td><Delta m={m} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted mt-2 text-[12.5px]">Green is better, red is worse. For error rate, latency and attack success, lower is better.</p>
      </Section>
      {!noKeys ? (
        <Section title="Cases that changed" id="cases" aside={<span className="text-[13px]">sign test p <strong>{c.sign_test_p === null ? dash : num(c.sign_test_p, 3)}</strong></span>}>
          <div className="grid gap-4 md:grid-cols-2">
            <CaseList title="Fixed" cases={c.fixed_cases ?? []} a={c.a.run_id} b={c.b.run_id} tone="pass" />
            <CaseList title="Regressed" cases={c.regressed_cases ?? []} a={c.a.run_id} b={c.b.run_id} tone="fail" />
          </div>
        </Section>
      ) : null}
      {c.only_in_a.length || c.only_in_b.length ? (
        <Section title="Cases in only one run" id="only">
          <div className="card card-pad grid gap-2 text-[14px]">
            {c.only_in_a.length ? <div>Only in A: {c.only_in_a.map((x) => <Chip key={x}>{x}</Chip>)}</div> : null}
            {c.only_in_b.length ? <div>Only in B: {c.only_in_b.map((x) => <Chip key={x}>{x}</Chip>)}</div> : null}
          </div>
        </Section>
      ) : null}
    </>
  );
}
