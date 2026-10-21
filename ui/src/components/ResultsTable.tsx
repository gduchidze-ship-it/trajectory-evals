import { ExternalLink } from "lucide-react";
import { Link } from "react-router-dom";
import type { ResultSummary } from "../api/types";
import { dateTime, dash, pct, shortRun } from "../lib/format";
import { resultHref } from "../lib/links";
import { Chip, kindLabel } from "./ui";

function VerdictChip({ r }: { r: ResultSummary }) {
  if (!r.complete) return <Chip tone="warn" title="The report has no verdict: still running, or interrupted.">Incomplete</Chip>;
  if (r.verdict === "adopt") return <Chip tone="pass">Adopt</Chip>;
  if (r.verdict) return <Chip tone="fail">{r.verdict}</Chip>;
  return null;
}

/** A pass rate with its interval as text and as a small bar, so it can be scanned down a column. */
export function PassCell({ r }: { r: ResultSummary }) {
  if (r.pass_rate === null) {
    return r.attack_succeeded !== null ? <span title="Attack success rate">ASR {pct(r.attack_succeeded, 1)}</span> : <span className="muted">{dash}</span>;
  }
  const lo = r.pass_rate_ci_low ?? r.pass_rate;
  const hi = r.pass_rate_ci_high ?? r.pass_rate;
  return (
    <div className="flex items-center justify-end gap-3">
      <svg width="64" height="12" viewBox="0 0 64 12" role="img" aria-label={`interval ${pct(lo)} to ${pct(hi)}`}>
        <rect x="0" y="5" width="64" height="2" rx="1" fill="var(--viz-grid)" />
        <rect x={lo * 62 + 1} y="4" width={Math.max(2, (hi - lo) * 62)} height="4" rx="2" fill="var(--viz-1)" opacity="0.4" />
        <circle cx={r.pass_rate * 62 + 1} cy="6" r="3.5" fill="var(--viz-1)" stroke="var(--surface)" strokeWidth="1.5" />
      </svg>
      <span className="w-[3.2ch] text-right font-medium">{pct(r.pass_rate)}</span>
    </div>
  );
}

export function ResultsTable({ items, showAgent = true }: { items: ResultSummary[]; showAgent?: boolean }) {
  return (
    <div className="card overflow-x-auto">
      <table className="tbl">
        <caption className="sr-only">Runs</caption>
        <thead>
          <tr>
            <th scope="col">Run</th>
            <th scope="col">Kind</th>
            {showAgent ? <th scope="col">Agent</th> : null}
            <th scope="col">When</th>
            <th scope="col" className="num">Pass rate (95% CI)</th>
            <th scope="col" className="num">Errors</th>
            <th scope="col" className="num">Rows</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {items.map((r) => (
            <tr key={r.run_id} className="row-link">
              <td className="max-w-[300px]">
                <Link to={resultHref(r.run_id)} className="font-medium [overflow-wrap:anywhere]">{shortRun(r.run_id)}</Link>
                <div className="muted text-[12px]">{r.dataset_file ?? r.dataset ?? dash}</div>
              </td>
              <td><Chip>{kindLabel[r.kind] ?? r.kind}</Chip></td>
              {showAgent ? <td>{r.agent ?? dash}</td> : null}
              <td className="whitespace-nowrap">{dateTime(r.created_at)}</td>
              <td className="num"><PassCell r={r} /></td>
              <td className="num">{pct(r.error_rate)}</td>
              <td className="num">{r.rows ?? dash}</td>
              <td>
                <div className="flex flex-wrap items-center gap-1.5">
                  {r.local === true ? <Chip title="Results stayed on this machine">local</Chip> : null}
                  <VerdictChip r={r} />
                  {r.url && /^https?:\/\//.test(r.url) ? (
                    <a href={r.url} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 text-[13px]">
                      LangSmith <ExternalLink size={12} aria-hidden="true" />
                    </a>
                  ) : null}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
