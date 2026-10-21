import { ArrowLeft, ArrowRight, CircleCheck, CircleX, Search, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { useEvalDetail, useResult, useRow, useRows } from "../api/hooks";
import type { ExperimentDetail, RowDetail, RowStatus, RowSummary } from "../api/types";
import { BarTable, ChartFrame, IntervalBar, Meter, ShareBars } from "../components/charts";
import { Transcript } from "../components/Transcript";
import { Banner, Chip, Drawer, EmptyState, ErrorState, Page, Section, SkeletonRows, StatusChip, kindLabel } from "../components/ui";
import { dash, dateTime, ms, num, pct, plural, shortRun } from "../lib/format";
import { useDebounced, useSlashFocus } from "../lib/hooks";
import { resultHref } from "../lib/links";
import { OptimizeReportView } from "./OptimizeReport";

export function ResultDetailPage() {
  const { runId = "", evalId } = useParams();
  const top = useResult(runId);
  const ev = useEvalDetail(runId, evalId);
  const q = evalId ? ev : top;

  if (q.isPending) return <Page title="Result"><SkeletonRows rows={6} label="Loading result" /></Page>;
  if (q.isError) {
    const gone = q.error instanceof ApiError && q.error.status === 404;
    return (
      <Page title={gone ? "Result not found" : "Result"} back={<Link to="/results">Results</Link>}>
        {gone ? <EmptyState title="No such run">{q.error.message} <Link to="/results">Back to results</Link>.</EmptyState> : <ErrorState error={q.error} onRetry={() => void q.refetch()} />}
      </Page>
    );
  }
  const d = q.data;
  if (d.detail === "optimize") return <OptimizeReportView detail={d} />;
  return <ExperimentView detail={d} runId={runId} {...(evalId ? { evalId } : {})} />;
}

function Kpi({ label, children, sub }: { label: string; children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="card card-pad min-w-0">
      <div className="eyebrow">{label}</div>
      <div className="mt-1 text-[26px] font-semibold leading-tight tabular-nums">{children}</div>
      {sub ? <div className="muted mt-0.5 text-[12.5px]">{sub}</div> : null}
    </div>
  );
}

const STATUSES: RowStatus[] = ["passed", "failed", "errored", "unscored"];

export function ExperimentView({ detail: d, runId, evalId }: { detail: ExperimentDetail; runId: string; evalId?: string }) {
  const [sp, setSp] = useSearchParams();
  const status = STATUSES.find((s) => s === sp.get("status")) ?? "";
  const segment = sp.get("segment") ?? "";
  const rowParam = sp.get("row");
  const openRow = rowParam !== null && /^\d+$/.test(rowParam) ? Number(rowParam) : null;
  const page = Math.max(0, Number(sp.get("page") ?? 0) || 0);
  const [search, setSearch] = useState(sp.get("q") ?? "");
  const q = useDebounced(search.trim(), 300);
  const searchRef = useRef<HTMLInputElement>(null);
  const opener = useRef<HTMLElement | null>(null);
  useSlashFocus(searchRef);
  const LIMIT = 50;

  const patch = (changes: Record<string, string | null>, keepPage = false) => {
    const n = new URLSearchParams(sp);
    for (const [k, v] of Object.entries(changes)) {
      if (v) n.set(k, v);
      else n.delete(k);
    }
    if (!keepPage) n.delete("page");
    setSp(n, { replace: true });
  };
  useEffect(() => {
    if ((sp.get("q") ?? "") !== q) patch({ q: q || null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q]);

  const rows = useRows(runId, { status, q, segment, offset: page * LIMIT, limit: LIMIT }, evalId);
  const counts = {
    all: useRows(runId, { limit: 1 }, evalId).data?.total,
    passed: useRows(runId, { status: "passed", limit: 1 }, evalId).data?.total,
    failed: useRows(runId, { status: "failed", limit: 1 }, evalId).data?.total,
    errored: useRows(runId, { status: "errored", limit: 1 }, evalId).data?.total,
    unscored: useRows(runId, { status: "unscored", limit: 1 }, evalId).data?.total,
  };

  const s = d.summary;
  const attack = d.personas.filter((p) => p.kind === "attack");
  const benign = d.personas.filter((p) => p.kind !== "attack");
  const lowCoverage = d.attack_succeeded_coverage !== null && d.attack_succeeded_coverage < 1;
  const segmentOptions = [
    ...Object.entries(d.segments).flatMap(([k, vals]) => Object.keys(vals).map((v) => `${k}=${v}`)),
    ...d.personas.map((p) => `persona=${p.persona}`),
  ].filter((v, i, a) => a.indexOf(v) === i);
  const defended = (() => {
    const w = attack.filter((p) => p.defended !== null);
    const tot = w.reduce((a, p) => a + p.rows * p.coverage, 0);
    return tot ? w.reduce((a, p) => a + (p.defended ?? 0) * p.rows * p.coverage, 0) / tot : null;
  })();

  const filtered = status !== "" || q !== "" || segment !== "";

  return (
    <Page
      title={<span className="[overflow-wrap:anywhere]">{shortRun(evalId ?? d.run_id)}</span>}
      back={evalId ? <Link to={resultHref(runId)}>Back to the report</Link> : <Link to="/results">Results</Link>}
      subtitle={<>{d.experiment ?? dash} · {d.dataset ?? dash} · {dateTime(d.created_at)}</>}
      actions={
        <>
          <Chip>{kindLabel[d.kind] ?? d.kind}</Chip>
          {d.agent ? <Chip>{d.agent}</Chip> : null}
          {d.local ? <Chip title="Results stayed on this machine">local</Chip> : null}
          {!evalId ? <Link className="btn" to={`/compare?a=${encodeURIComponent(runId)}`}>Compare with another run</Link> : null}
        </>
      }
    >
      {d.pass_keys === null ? (
        <Banner tone="warn">This run did not record its pass keys, so its rows cannot be marked passed or failed and show as unscored. Re-run it to get pass and fail.</Banner>
      ) : null}

      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {d.pass_rate !== null || d.kind !== "redteam" ? (
          <div className="card card-pad sm:col-span-2">
            <div className="eyebrow">Pass rate</div>
            <div className="flex flex-wrap items-end gap-x-5 gap-y-1">
              <div className="mt-1 text-[32px] font-semibold leading-none tabular-nums">{pct(d.pass_rate, 1)}</div>
              <div className="muted pb-0.5 text-[13px]">95% CI {pct(d.pass_rate_ci_low, 1)} to {pct(d.pass_rate_ci_high, 1)} (Wilson)</div>
            </div>
            <div className="mt-2"><IntervalBar value={d.pass_rate} low={d.pass_rate_ci_low} high={d.pass_rate_ci_high} label="Pass rate" /></div>
          </div>
        ) : null}
        {d.attack_succeeded !== null ? (
          <Kpi label="Attack success rate" sub={`coverage ${pct(d.attack_succeeded_coverage)}`}>{pct(d.attack_succeeded, 1)}</Kpi>
        ) : null}
        {defended !== null ? <Kpi label="Defended rate" sub="attack rows scored">{pct(defended, 1)}</Kpi> : null}
        <Kpi label="Error rate" sub={plural(d.rows ?? 0, "row")}>{pct(d.error_rate, 1)}</Kpi>
        <Kpi label="Latency p50" sub={`p95 ${ms(s["latency_p95_ms"])}`}>{ms(s["latency_p50_ms"])}</Kpi>
        {s["pass_all_k"] !== undefined && s["pass_all_k"] !== null ? <Kpi label="Pass all k" sub={`pass any k ${pct(s["pass_any_k"] ?? null)}`}>{pct(s["pass_all_k"])}</Kpi> : null}
      </div>

      {lowCoverage ? (
        <Banner tone="warn">
          <strong>The attack success rate is uncertain.</strong> attack_succeeded scored {pct(d.attack_succeeded_coverage, 1)} of the rows, not all. Errored attack rows count as neither defended nor broken.
        </Banner>
      ) : null}

      {attack.length > 0 ? (
        <Section title="Attack success by persona" id="personas">
          <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            <ChartFrame
              title="Attack success rate (lower is better)"
              caption="Share of scored attack rows where the attack succeeded"
              table={<BarTable valueHeader="ASR" data={attack.map((p) => ({ name: p.persona, value: p.attack_succeeded }))} />}
            >
              <ShareBars ariaLabel="Attack success rate per persona" data={attack.map((p) => ({ name: p.persona, value: p.attack_succeeded, note: `${pct(p.coverage)} scored, ${plural(p.errors, "error")}` }))} />
            </ChartFrame>
            <div className="card overflow-x-auto">
              <table className="tbl">
                <caption className="sr-only">Attack personas</caption>
                <thead><tr><th>Persona</th><th className="num">Rows</th><th className="num">ASR</th><th className="num">Defended</th><th className="num">Coverage</th><th className="num">Turns to break</th><th className="num">Errors</th><th><span className="sr-only">Filter</span></th></tr></thead>
                <tbody>
                  {attack.map((p) => (
                    <tr key={p.persona}>
                      <td className="font-medium">{p.persona}</td>
                      <td className="num">{p.rows}</td>
                      <td className="num">{pct(p.attack_succeeded)}</td>
                      <td className="num">{pct(p.defended)}</td>
                      <td className="num">{p.coverage < 1 ? <span className="inline-flex items-center gap-1 text-warn"><TriangleAlert size={13} aria-hidden="true" />{pct(p.coverage)}</span> : pct(p.coverage)}</td>
                      <td className="num">{num(p.turns_to_break, 1)}</td>
                      <td className="num">{p.errors}</td>
                      <td><button className="btn btn-sm" onClick={() => patch({ segment: `persona=${p.persona}` })} aria-label={`Show rows for ${p.persona}`}>Rows</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Section>
      ) : null}

      {benign.length > 0 ? (
        <Section title="Benign personas" id="benign">
          <div className="card overflow-x-auto">
            <table className="tbl">
              <caption className="sr-only">Benign personas</caption>
              <thead><tr><th>Persona</th><th className="num">Rows</th><th className="num">Errors</th><th className="num">Coverage</th><th><span className="sr-only">Filter</span></th></tr></thead>
              <tbody>
                {benign.map((p) => (
                  <tr key={p.persona}>
                    <td className="font-medium">{p.persona}</td>
                    <td className="num">{p.rows}</td>
                    <td className="num">{p.errors}</td>
                    <td className="num">{pct(p.coverage)}</td>
                    <td><button className="btn btn-sm" onClick={() => patch({ segment: `persona=${p.persona}` })} aria-label={`Show rows for ${p.persona}`}>Rows</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      ) : null}

      <div className="mb-6 grid items-start gap-4 lg:grid-cols-2">
        <Section title="Graders">
          {d.feedback.length === 0 ? (
            <div className="card card-pad muted text-[14px]">No grader scores in this run.</div>
          ) : (
            <div className="card overflow-x-auto">
              <table className="tbl">
                <caption className="sr-only">Mean score and coverage per grader</caption>
                <thead><tr><th>Grader</th><th className="num">Mean</th><th>Coverage</th></tr></thead>
                <tbody>
                  {d.feedback.map((f) => {
                    const low = f.coverage !== null && f.coverage < 1;
                    return (
                      <tr key={f.key}>
                        <td className="mono text-[13px]">{f.key}</td>
                        <td className="num">{pct(f.mean)}</td>
                        <td>
                          <div className="flex items-center gap-2">
                            <Meter value={f.coverage} warn={low} label={`${f.key} coverage`} />
                            <span className={low ? "inline-flex items-center gap-1 whitespace-nowrap font-medium text-warn" : "whitespace-nowrap"}>
                              {low ? <TriangleAlert size={13} aria-label="partial coverage" /> : null}{pct(f.coverage)}
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Section>
        {Object.entries(d.segments).map(([key, vals]) => {
          const data = Object.entries(vals).map(([name, value]) => ({ name, value }));
          return (
            <Section key={key} title={`Pass rate by ${key}`}>
              <ChartFrame title={`Pass rate by ${key}`} table={<BarTable valueHeader="Pass rate" data={data} />}>
                <ShareBars ariaLabel={`Pass rate by ${key}`} data={data} />
              </ChartFrame>
            </Section>
          );
        })}
      </div>

      <Section title="Rows" id="rows">
        <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          <div role="group" aria-label="Filter by status" className="flex flex-wrap gap-1.5">
            {(["", ...STATUSES] as const).map((st) => {
              const n = counts[st === "" ? "all" : st];
              const on = status === st;
              return (
                <button key={st || "all"} type="button" aria-pressed={on} className={`btn btn-sm ${on ? "border-accent bg-accent-soft text-accent-ink" : ""}`} onClick={() => patch({ status: st || null })}>
                  {st === "" ? "All" : st[0]!.toUpperCase() + st.slice(1)}
                  <span className="muted tabular-nums">{n ?? "·"}</span>
                </button>
              );
            })}
          </div>
          <div className="relative min-w-[200px] flex-1 sm:max-w-[280px]">
            <Search size={15} className="pointer-events-none absolute left-3 top-[10px] text-ink-3" aria-hidden="true" />
            <input ref={searchRef} type="search" className="input pl-9" placeholder="Search case id" aria-label="Search case id" value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>
          {segmentOptions.length ? (
            <select className="input w-auto" aria-label="Filter by segment" value={segment} onChange={(e) => patch({ segment: e.target.value || null })}>
              <option value="">All segments</option>
              {segmentOptions.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          ) : null}
          {filtered ? <button className="btn btn-sm btn-ghost" onClick={() => { setSearch(""); patch({ status: null, q: null, segment: null }); }}>Clear filters</button> : null}
        </div>

        {rows.isPending ? (
          <SkeletonRows rows={5} label="Loading rows" />
        ) : rows.isError ? (
          <ErrorState error={rows.error} onRetry={() => void rows.refetch()} title="Could not load rows" />
        ) : rows.data.total === 0 ? (
          <EmptyState title={filtered ? "No rows match these filters" : "This run has no rows"} action={filtered ? <button className="btn" onClick={() => { setSearch(""); patch({ status: null, q: null, segment: null }); }}>Clear filters</button> : undefined} />
        ) : (
          <>
            {rows.data.items.length ? <RowsTable items={rows.data.items} onOpen={(i, el) => { opener.current = el; patch({ row: String(i) }, true); }} selected={openRow} /> : <EmptyState title="No rows on this page">Go back to an earlier page.</EmptyState>}
            {rows.data.total > LIMIT ? (
              <div className="mt-3 flex items-center justify-end gap-2 text-[13px]">
                <button className="btn btn-sm" disabled={page === 0} onClick={() => patch({ page: String(page - 1) }, true)}>Previous</button>
                <span aria-live="polite">{page * LIMIT + 1} to {Math.min(rows.data.total, (page + 1) * LIMIT)} of {rows.data.total}</span>
                <button className="btn btn-sm" disabled={(page + 1) * LIMIT >= rows.data.total} onClick={() => patch({ page: String(page + 1) }, true)}>Next</button>
              </div>
            ) : null}
          </>
        )}
      </Section>

      <RowDrawer
        runId={runId}
        {...(evalId ? { evalId } : {})}
        index={openRow}
        order={rows.data?.items.map((r) => r.index) ?? []}
        onIndex={(i) => patch({ row: i === null ? null : String(i) }, true)}
        hasPassKeys={d.pass_keys !== null}
        restoreFocus={() => opener.current?.focus()}
      />
    </Page>
  );
}

export function FeedbackChips({ feedback, hideKey }: { feedback: RowSummary["feedback"]; hideKey?: boolean }) {
  const entries = Object.entries(feedback);
  if (!entries.length) return <span className="muted">{dash}</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {entries.map(([k, v]) => {
        // attack_succeeded is the one boolean where true is the bad outcome.
        const good = k === "attack_succeeded" ? v === false : v === true;
        const bad = k === "attack_succeeded" ? v === true : v === false;
        return (
        <span key={k} className={`chip ${good ? "chip-pass" : bad ? "chip-fail" : ""}`} title={`${k}: ${v === null ? "not scored" : String(v)}`}>
          {good ? <CircleCheck size={11} aria-hidden="true" /> : bad ? <CircleX size={11} aria-hidden="true" /> : null}
          {hideKey ? (typeof v === "number" ? num(v) : v === null ? "n/a" : String(v)) : k}
          {!hideKey ? (typeof v === "number" ? ` ${num(v)}` : v === null ? " n/a" : "") : ""}
        </span>
        );
      })}
    </span>
  );
}

function RowsTable({ items, onOpen, selected }: { items: RowSummary[]; onOpen: (i: number, el: HTMLElement) => void; selected: number | null }) {
  return (
    <div className="card overflow-x-auto">
      <table className="tbl">
        <caption className="sr-only">Rows. Press Enter on a row to open its transcript.</caption>
        <thead><tr><th scope="col">Case</th><th scope="col">Status</th><th scope="col">Input</th><th scope="col">Reply</th><th scope="col">Graders</th><th scope="col" className="num">Latency</th></tr></thead>
        <tbody>
          {items.map((r) => (
            <tr
              key={r.index}
              className="row-link cursor-pointer"
              tabIndex={0}
              data-selected={selected === r.index}
              aria-current={selected === r.index ? "true" : undefined}
              onClick={(e) => onOpen(r.index, e.currentTarget)}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(r.index, e.currentTarget); } }}
            >
              <td className="max-w-[220px]"><div className="mono truncate text-[13px] font-medium">{r.case_id}</div>{r.persona ? <div className="muted text-[12px]">{r.persona}</div> : null}</td>
              <td><StatusChip status={r.status} /></td>
              <td className="max-w-[240px] truncate">{r.input_preview}</td>
              <td className="max-w-[240px] truncate">{r.error ? <span className="text-errored">{r.error}</span> : r.reply_preview}</td>
              <td className="max-w-[260px]"><FeedbackChips feedback={r.feedback} /></td>
              <td className="num">{ms(r.latency_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RowDrawer({ runId, evalId, index, order, onIndex, hasPassKeys, restoreFocus }: { runId: string; evalId?: string; index: number | null; order: number[]; onIndex: (i: number | null) => void; hasPassKeys: boolean; restoreFocus: () => void }) {
  const q = useRow(runId, index, evalId);
  const pos = index === null ? -1 : order.indexOf(index);
  const prev = pos > 0 ? order[pos - 1] : undefined;
  const next = pos >= 0 ? order[pos + 1] : undefined;

  useEffect(() => {
    if (index === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "j" && next !== undefined) onIndex(next);
      if (e.key === "k" && prev !== undefined) onIndex(prev);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, next, prev, onIndex]);

  return (
    <Drawer open={index !== null} onOpenChange={(o) => !o && onIndex(null)} title={q.data?.case_id ?? "Row"} description="Transcript and grader reasons" restoreFocus={restoreFocus}>
      {q.isPending ? <SkeletonRows rows={5} label="Loading row" /> : q.isError ? <ErrorState error={q.error} onRetry={() => void q.refetch()} /> : <RowBody row={q.data} hasPassKeys={hasPassKeys} />}
      <div className="mt-5 flex items-center justify-between border-t border-line pt-3">
        <button className="btn btn-sm" disabled={prev === undefined} onClick={() => prev !== undefined && onIndex(prev)}><ArrowLeft size={13} aria-hidden="true" /> Previous <span className="kbd">k</span></button>
        <button className="btn btn-sm" disabled={next === undefined} onClick={() => next !== undefined && onIndex(next)}>Next <span className="kbd">j</span> <ArrowRight size={13} aria-hidden="true" /></button>
      </div>
    </Drawer>
  );
}

function RowBody({ row, hasPassKeys }: { row: RowDetail; hasPassKeys: boolean }) {
  const messages = useMemo(() => {
    const m = row.outputs["messages"];
    if (Array.isArray(m) && m.length) return m;
    const fallback: unknown[] = Array.isArray(row.inputs["messages"]) ? [...(row.inputs["messages"] as unknown[])] : [];
    if (typeof row.outputs["text"] === "string") fallback.push({ role: "assistant", content: row.outputs["text"] });
    return fallback;
  }, [row]);
  const meta = Object.entries(row.metadata);
  return (
    <div className="grid gap-5">
      <div className="flex flex-wrap items-center gap-2">
        <StatusChip status={row.status} />
        {!hasPassKeys ? <Chip title="The run did not record its pass keys">pass keys not recorded</Chip> : null}
        {row.persona ? <Chip tone="accent">{row.persona}</Chip> : null}
        {meta.map(([k, v]) => <Chip key={k}>{k}: {String(v)}</Chip>)}
        <span className="muted ml-auto text-[13px]">{ms(row.latency_ms)}</span>
      </div>
      {row.error ? <p role="alert" className="pre-text m-0 rounded-md bg-errored-soft px-3 py-2 text-[13.5px] text-errored">{row.error}</p> : null}
      <section aria-labelledby="graders-h">
        <h3 id="graders-h" className="eyebrow mb-2">Graders</h3>
        {Object.keys(row.feedback).length === 0 ? <p className="muted m-0 text-[14px]">No grader scored this row.</p> : (
          <ul className="m-0 grid list-none gap-2 p-0">
            {Object.entries(row.feedback).map(([k, v]) => (
              <li key={k} className="rounded-lg border border-line px-3 py-2">
                <div className="flex items-center gap-2"><span className="mono text-[13px] font-medium">{k}</span><FeedbackChips feedback={{ [k]: v }} hideKey /></div>
                {row.comments[k] ? <p className="pre-text m-0 mt-1 text-[13.5px]"><span className="muted">Reason: </span>{row.comments[k]}</p> : null}
              </li>
            ))}
          </ul>
        )}
        {row.feedback_errors.length ? (
          <ul className="m-0 mt-2 list-none rounded-md bg-fail-soft p-3 text-[13px] text-fail">
            {row.feedback_errors.map((e, i) => <li key={i} className="pre-text flex gap-2"><TriangleAlert size={14} className="mt-0.5 shrink-0" aria-hidden="true" /> {e}</li>)}
          </ul>
        ) : null}
      </section>
      <section aria-labelledby="tx-h">
        <h3 id="tx-h" className="eyebrow mb-2">Transcript</h3>
        <Transcript messages={messages} />
      </section>
      {Object.keys(row.reference_outputs).length ? (
        <details>
          <summary className="cursor-pointer text-[14px] font-medium">Reference outputs</summary>
          <pre className="pre-text mt-2 rounded-md bg-sunken p-3 text-[12.5px]">{JSON.stringify(row.reference_outputs, null, 2)}</pre>
        </details>
      ) : null}
    </div>
  );
}
