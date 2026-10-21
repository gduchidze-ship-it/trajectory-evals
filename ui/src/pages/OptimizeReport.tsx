import { CircleCheck, CircleDashed, CircleX, GitBranch, Trophy } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { usePrompts } from "../api/hooks";
import type { CandidateRecord, Holdout, OptimizeDetail } from "../api/types";
import { BarTable, ChartFrame, ShareBars } from "../components/charts";
import { CodeDiff } from "../components/CodeEditor";
import { Banner, Chip, ErrorState, Page, Section, SkeletonRows } from "../components/ui";
import { evaluateFloors, lineage } from "../lib/floors";
import { dash, isShare, num, pct, plural, signed } from "../lib/format";
import { resultHref } from "../lib/links";

function verdictTone(v: string): "pass" | "fail" | "warn" | "accent" | undefined {
  if (v === "accepted" || v === "passed floors" || v === "adopt") return "pass";
  if (v === "rejected" || v === "do not adopt") return "fail";
  if (v === "invalid") return "warn";
  if (v === "baseline") return "accent";
  return undefined;
}

function Delta({ v, lowerIsBetter, share = true }: { v: number | null; lowerIsBetter?: boolean; share?: boolean }) {
  if (v === null) return <span className="muted">{dash}</span>;
  const good = lowerIsBetter ? v < 0 : v > 0;
  const bad = lowerIsBetter ? v > 0 : v < 0;
  return <span className={good ? "text-pass" : bad ? "text-fail" : "muted"}>{signed(v, share)}</span>;
}

export function OptimizeReportView({ detail }: { detail: OptimizeDetail }) {
  const rep = detail.report;
  const records = rep.mode === "optimize" ? rep.candidates : rep.variants;
  const nodes = useMemo(() => lineage(records), [records]);
  const winner = rep.mode === "optimize" ? rep.best : rep.winner;
  const [diffId, setDiffId] = useState<string | null>(() => (winner && winner !== "baseline" ? winner : records.find((r) => r.verdict !== "baseline")?.id ?? null));
  const valBars = records.map((r) => ({ name: r.id, value: r.val?.["pass_rate"] ?? null }));
  const incomplete = !detail.complete || rep.verdict === null;

  return (
    <Page
      title={`${rep.mode === "optimize" ? "Optimizer" : "Sweep"} report`}
      back={<Link to="/results">Results</Link>}
      subtitle={<span className="mono text-[12.5px]">{detail.run_id}</span>}
      actions={
        <>
          <Chip>{rep.agent}</Chip>
          {rep.upload ? <Chip>uploaded</Chip> : <Chip title="Results stayed on this machine">local</Chip>}
        </>
      }
    >
      {incomplete ? <Banner tone="warn">In progress or interrupted: this report has no verdict yet. Candidates so far are shown below; the holdout check runs last.</Banner> : null}

      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card card-pad sm:col-span-2">
          <div className="eyebrow">Verdict</div>
          <div className="mt-1 flex flex-wrap items-center gap-2">
            {rep.verdict ? <span className={`chip text-[15px] ${verdictTone(rep.verdict) ? `chip-${verdictTone(rep.verdict)}` : ""}`}>{rep.verdict === "adopt" ? <CircleCheck size={15} aria-hidden="true" /> : <CircleX size={15} aria-hidden="true" />}{rep.verdict}</span> : <Chip tone="warn">no verdict yet</Chip>}
            {winner ? <span className="inline-flex items-center gap-1 text-[14px]"><Trophy size={14} aria-hidden="true" /> {rep.mode === "optimize" ? "best" : "winner"}: <strong>{winner}</strong></span> : null}
          </div>
          {rep.reason ? <p className="pre-text m-0 mt-2 text-[13.5px]">{rep.reason}</p> : null}
        </div>
        <div className="card card-pad">
          <div className="eyebrow">Splits</div>
          <div className="mt-1 text-[14px] tabular-nums">{rep.splits.train} train · {rep.splits.val} val · {rep.splits.holdout} holdout</div>
          <div className="muted text-[12.5px]">{rep.splits.labelled ? "labelled by the dataset" : "split by hash"}</div>
        </div>
        <div className="card card-pad">
          <div className="eyebrow">Agent calls</div>
          <div className="mt-1 text-[22px] font-semibold tabular-nums">{rep.agent_calls}</div>
          {rep.mode === "optimize" && rep.stop_reason ? <div className="muted text-[12.5px]">stopped: {rep.stop_reason}</div> : null}
        </div>
      </div>

      <Section title={rep.mode === "optimize" ? "Candidate lineage" : "Variants"} id="lineage">
        <ol className="m-0 grid list-none gap-2 p-0" aria-label={rep.mode === "optimize" ? "Candidate lineage" : "Variants"}>
          {nodes.map(({ record: c, depth, parentVal }) => (
            <li key={c.id} style={{ marginLeft: Math.min(depth, 6) * 24 }} data-testid={`candidate-${c.id}`}>
              <CandidateCard c={c} depth={depth} parentVal={parentVal} optimize={rep.optimize} isWinner={c.id === winner} runId={detail.run_id} evals={detail.evals} onDiff={() => setDiffId(c.id)} diffing={diffId === c.id} sweep={rep.mode === "sweep"} />
            </li>
          ))}
        </ol>
      </Section>

      <div className="mb-6">
        <ChartFrame title="Validation pass rate by candidate" caption="Higher is better" table={<BarTable valueHeader="Val pass rate" data={valBars} />}>
          <ShareBars data={valBars} ariaLabel="Validation pass rate by candidate" />
        </ChartFrame>
      </div>

      {diffId ? <PromptDiff runId={detail.run_id} record={records.find((r) => r.id === diffId)} /> : null}

      <Section title="Holdout" id="holdout">
        {rep.holdout ? <HoldoutPanel h={rep.holdout} verdict={rep.verdict} /> : <div className="card card-pad muted text-[14px]">The holdout check has not run: it happens after the candidates are done.</div>}
      </Section>

      <Section title="Experiments" id="evals">
        {detail.evals.length ? (
          <ul className="m-0 flex list-none flex-wrap gap-2 p-0">
            {detail.evals.map((e) => <li key={e}><Link className="btn btn-sm" to={resultHref(detail.run_id, e)}>{e}</Link></li>)}
          </ul>
        ) : <p className="muted text-[14px]">No experiment folders yet.</p>}
      </Section>
    </Page>
  );
}

function CandidateCard({ c, depth, parentVal, optimize, isWinner, runId, evals, onDiff, diffing, sweep }: {
  c: CandidateRecord;
  depth: number;
  parentVal: number | null;
  optimize: Record<string, unknown>;
  isWinner: boolean;
  runId: string;
  evals: string[];
  onDiff: () => void;
  diffing: boolean;
  sweep: boolean;
}) {
  const val = c.val?.["pass_rate"] ?? null;
  const delta = val !== null && parentVal !== null ? val - parentVal : c.deltas["pass_rate_delta_vs_previous"] ?? null;
  const floors = evaluateFloors(optimize, c);
  const evalId = evals.find((e) => e === `${c.id}-val`);
  return (
    <article className={`card card-pad ${isWinner ? "border-accent" : ""} ${diffing ? "outline outline-2 outline-offset-2" : ""}`} style={diffing ? { outlineColor: "var(--accent)" } : undefined}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {depth > 0 ? <GitBranch size={14} className="text-ink-3" aria-hidden="true" /> : null}
        <h3 className="mono text-[14px]">{c.id}</h3>
        <Chip tone={verdictTone(c.verdict)}>{c.verdict}</Chip>
        {isWinner ? <Chip tone="accent"><Trophy size={11} aria-hidden="true" /> {sweep ? "winner" : "best"}</Chip> : null}
        <span className="ml-auto flex items-center gap-4 text-[14px] tabular-nums">
          {c.cost !== null ? <span className="muted">cost {num(c.cost, 2)}</span> : null}
          <span>val pass rate <strong>{pct(val, 1)}</strong>{c.parent ? <> <span className="muted">vs {c.parent}:</span> <Delta v={delta} /></> : null}</span>
        </span>
      </div>
      {c.rationale ? <p className="pre-text m-0 mt-2 text-[13.5px]"><span className="muted">Change: </span>{c.rationale}</p> : null}
      {c.reason ? <p className="pre-text m-0 mt-1 text-[13.5px] text-fail"><span className="font-medium">Why not: </span>{c.reason}</p> : null}
      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-[13px]">
        <span title={c.fixed_cases.join(", ")}>fixed <strong>{c.fixed_cases.length}</strong></span>
        <span title={c.regressed_cases.join(", ")} className={c.regressed_cases.length ? "text-fail" : ""}>regressed <strong>{c.regressed_cases.length}</strong></span>
        <span>sign test p <strong>{c.sign_test_p === null ? dash : num(c.sign_test_p, 3)}</strong></span>
        <span className="muted">{plural(c.agent_calls, "agent call")}</span>
        {c.redteam ? <span>ASR <strong>{pct(c.redteam["attack_succeeded"] ?? null, 1)}</strong></span> : null}
      </div>
      {c.fixed_cases.length + c.regressed_cases.length > 0 ? (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {c.fixed_cases.map((x) => <Chip key={`f${x}`} tone="pass">fixed: {x}</Chip>)}
          {c.regressed_cases.map((x) => <Chip key={`r${x}`} tone="fail">regressed: {x}</Chip>)}
        </div>
      ) : null}
      {floors.length ? (
        <ul className="m-0 mt-2 flex list-none flex-wrap gap-1.5 p-0" aria-label={`Floors for ${c.id}`}>
          {floors.map((f) => (
            <li key={f.key} className={`chip ${f.status === "pass" ? "chip-pass" : f.status === "fail" ? "chip-fail" : ""}`} title={`${f.label}: ${f.rule} ${f.threshold}; measured ${f.measured === null ? "nothing" : f.measured}`}>
              {f.status === "pass" ? <CircleCheck size={12} aria-label="floor passed" /> : f.status === "fail" ? <CircleX size={12} aria-label="floor failed" /> : <CircleDashed size={12} aria-label="floor not measured" />}
              {f.label} {f.rule} {f.threshold}, {f.measured === null ? "not measured" : `got ${num(f.measured, 3)}`}
            </li>
          ))}
        </ul>
      ) : null}
      <div className="mt-2 flex gap-2">
        {c.verdict !== "baseline" ? <button type="button" className="btn btn-sm" onClick={onDiff} aria-pressed={diffing}>Prompt diff vs {c.parent ?? "baseline"}</button> : null}
        {evalId ? <Link className="btn btn-sm" to={resultHref(runId, evalId)}>Open val experiment</Link> : null}
      </div>
    </article>
  );
}

function PromptDiff({ runId, record }: { runId: string; record: CandidateRecord | undefined }) {
  // The backend always resolves the "baseline" prompts, so a parentless variant diffs against them.
  const parentId = record?.parent ?? (record?.id !== "baseline" ? "baseline" : null);
  const child = usePrompts(runId, record?.id ?? null);
  const parent = usePrompts(runId, parentId);
  if (!record) return null;
  const names = child.data ? Object.keys(child.data) : [];
  return (
    <Section title={`Prompt diff: ${record.id} against ${parentId ?? "nothing"}`} id="diff">
      {child.isPending || (parentId !== null && parent.isPending) ? (
        <SkeletonRows rows={4} label="Loading prompts" />
      ) : child.isError ? (
        <ErrorState error={child.error} onRetry={() => void child.refetch()} title="Could not load this candidate's prompts" />
      ) : parent.isError ? (
        <ErrorState error={parent.error} onRetry={() => void parent.refetch()} title="Could not load the parent's prompts" />
      ) : names.length === 0 ? (
        <div className="card card-pad muted text-[14px]">This candidate has no prompt files.</div>
      ) : (
        <div className="grid gap-4">
          {names.map((n) => (
            <div key={n}>
              <div className="mono mb-1 text-[13px] font-medium">{n}.md</div>
              <CodeDiff label={`Diff of ${n} prompt`} original={parent.data?.[n] ?? ""} modified={child.data?.[n] ?? ""} />
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

function HoldoutPanel({ h, verdict }: { h: Holdout; verdict: string | null }) {
  const keys = [...new Set([...Object.keys(h.baseline), ...Object.keys(h.best)])].filter((k) => !k.endsWith("_coverage") && k !== "rows");
  const share = isShare;
  return (
    <div className="card card-pad grid gap-4">
      <div className="flex flex-wrap items-center gap-3 text-[14px]">
        <span>Candidate <strong className="mono">{h.candidate}</strong> against the baseline, on cases neither was tuned on.</span>
        {verdict ? <Chip tone={verdictTone(verdict)}>{verdict}</Chip> : null}
      </div>
      <div className="overflow-x-auto">
        <table className="tbl">
          <caption className="sr-only">Holdout summary, baseline against best</caption>
          <thead><tr><th>Metric</th><th className="num">Baseline</th><th className="num">{h.candidate}</th><th className="num">Change</th></tr></thead>
          <tbody>
            {keys.map((k) => {
              const a = h.baseline[k] ?? null;
              const b = h.best[k] ?? null;
              return (
                <tr key={k}>
                  <td className="mono text-[13px]">{k}</td>
                  <td className="num">{share(k) ? pct(a, 1) : num(a, 3)}</td>
                  <td className="num">{share(k) ? pct(b, 1) : num(b, 3)}</td>
                  <td className="num"><Delta v={a !== null && b !== null ? b - a : (h.deltas[k] ?? null)} lowerIsBetter={k.includes("error") || k.includes("latency") || k.includes("attack")} share={share(k)} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap gap-x-8 gap-y-2 text-[14px]">
        <div>Sign test p <strong className="tabular-nums">{num(h.sign_test_p, 3)}</strong> <span className="muted text-[12.5px]">over the cases whose result changed</span></div>
        <div className="flex flex-wrap items-center gap-1.5">Fixed {h.fixed_cases.length ? h.fixed_cases.map((x) => <Chip key={x} tone="pass">{x}</Chip>) : <strong>0</strong>}</div>
        <div className="flex flex-wrap items-center gap-1.5">Regressed {h.regressed_cases.length ? h.regressed_cases.map((x) => <Chip key={x} tone="fail">{x}</Chip>) : <strong>0</strong>}</div>
      </div>
    </div>
  );
}
