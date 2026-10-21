import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleCheck, Plus, Save, Trash2, Undo2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api, describeError } from "../api/client";
import type { DatasetKind, DatasetPage, RowError } from "../api/types";
import { CodeEditor } from "../components/CodeEditor";
import { Banner, Chip, EmptyState, ErrorState, Page, SkeletonRows, Spinner } from "../components/ui";
import { useDebounced } from "../lib/hooks";
import { agentHref } from "../lib/links";
import { basename, plural } from "../lib/format";

type Row = Record<string, unknown> | null;
const PAGE = 100;

interface Loaded {
  kind: DatasetKind;
  rows: Row[];
  errors: RowError[];
}

async function loadWhole(path: string): Promise<Loaded> {
  const rows: Row[] = [];
  let first: DatasetPage | null = null;
  for (let offset = 0; ; offset += 500) {
    const p = await api.dataset(path, { offset, limit: 500 });
    first ??= p;
    rows.push(...p.items.map((i) => i.data));
    if (rows.length >= p.total || p.items.length === 0) break;
  }
  return { kind: first?.kind ?? "golden", rows, errors: first?.errors ?? [] };
}

export const rowText = (v: unknown): string => (typeof v === "string" ? v : "");

/** The visible gist of a row: input, else the last user message, else its scenario goal. Text only. */
export function rowPreview(d: Record<string, unknown> | null): string {
  if (!d) return "";
  if (typeof d["input"] === "string") return d["input"];
  const m = d["messages"];
  if (Array.isArray(m)) {
    for (let i = m.length - 1; i >= 0; i--) {
      const x: unknown = m[i];
      if (typeof x === "object" && x !== null) {
        const o = x as Record<string, unknown>;
        if (o["role"] === "user" && typeof o["content"] === "string") return o["content"];
      }
      if (typeof x === "string") return x;
    }
  }
  return rowText(d["goal"]);
}

const sameRows = (a: Row[], b: Row[]): boolean => JSON.stringify(a) === JSON.stringify(b);

export function DatasetEditorPage() {
  const splat = useParams()["*"] ?? "";
  const path = splat.split("/").map(decodeURIComponent).join("/");
  const [sp] = useSearchParams();
  const agent = sp.get("agent");
  const qc = useQueryClient();
  const summaries = useQuery({ queryKey: ["datasets", { agent: null }], queryFn: () => api.datasets() });
  const whole = useQuery({ queryKey: ["datasets", path, "whole"], queryFn: () => loadWhole(path) });
  const meta = summaries.data?.find((d) => d.path === path);

  const [rows, setRows] = useState<Row[] | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [page, setPage] = useState(0);
  const [errors, setErrors] = useState<RowError[]>([]);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [rowJsonError, setRowJsonError] = useState<string | null>(null);
  const [rev, setRev] = useState(0);
  const baseline = useRef<Row[]>([]);

  useEffect(() => {
    if (whole.data) {
      setRows(whole.data.rows);
      baseline.current = whole.data.rows;
      setErrors(whole.data.errors);
    }
  }, [whole.data]);

  const kind = whole.data?.kind ?? "golden";
  const writable = meta?.writable ?? false;
  const dirty = rows !== null && !sameRows(rows, baseline.current);
  const debouncedRows = useDebounced(rows, 400);

  useEffect(() => {
    if (!debouncedRows || !dirty || debouncedRows.some((r) => r === null)) return;
    const ctl = new AbortController();
    api
      .validateRows({ rows: debouncedRows as Record<string, unknown>[], kind }, ctl.signal)
      .then((r) => setErrors(r.errors))
      .catch(() => undefined);
    return () => ctl.abort();
  }, [debouncedRows, dirty, kind]);

  const update = useCallback((i: number, next: Row) => setRows((r) => (r ? r.map((x, j) => (j === i ? next : x)) : r)), []);
  const errorsByLine = useMemo(() => new Map(errors.map((e) => [e.line, e.message])), [errors]);

  if (whole.isPending) return <Page title={basename(path)}><SkeletonRows rows={8} label="Loading dataset" /></Page>;
  if (whole.isError || rows === null) {
    return (
      <Page title={basename(path)} back={<Link to="/agents">Agents</Link>}>
        <ErrorState error={whole.error} onRetry={() => void whole.refetch()} title="Could not load this dataset" />
      </Page>
    );
  }

  const reviewCount = rows.filter((r) => r?.["review_required"] === true).length;
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const shown = rows.slice(page * PAGE, page * PAGE + PAGE);
  const sel = selected !== null ? rows[selected] ?? null : null;

  const onSave = async () => {
    if (rows.some((r) => r === null)) {
      setSaveError("A line is not a JSON object. Fix it in the side panel, or delete the row, before saving.");
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      await api.writeDataset(path, { rows: rows as Record<string, unknown>[], kind });
      baseline.current = rows;
      setSaved(true);
      window.setTimeout(() => setSaved(false), 3000);
      await qc.invalidateQueries({ queryKey: ["datasets"] });
    } catch (e) {
      setSaveError(describeError(e));
    } finally {
      setSaving(false);
    }
  };

  const approve = (i: number, expectedOutput?: string) => {
    const r = rows[i];
    if (!r) return;
    const observed = typeof r["observed_output"] === "string" ? r["observed_output"] : "";
    const expected = { ...((typeof r["expected"] === "object" && r["expected"] !== null ? r["expected"] : {}) as Record<string, unknown>), output: expectedOutput ?? observed };
    update(i, { ...r, expected, review_required: false });
  };

  const addRow = () => {
    const n = rows.length + 1;
    setRows([...rows, kind === "scenarios" ? { id: `new-${n}`, persona: "", goal: "" } : { id: `new-${n}`, input: "", expected: {} }]);
    setPage(Math.floor(rows.length / PAGE));
    setSelected(rows.length);
  };

  return (
    <Page
      title={<span className="mono text-[20px]">{basename(path)}</span>}
      back={agent ? <Link to={agentHref(agent, "datasets")}>Back to agent</Link> : <Link to="/agents">Agents</Link>}
      subtitle={<span className="mono text-[12.5px]">{path}</span>}
      actions={
        <>
          <Chip>{kind}</Chip>
          <Chip>{plural(rows.length, "row")}</Chip>
          {errors.length ? <Chip tone="fail">{plural(errors.length, "problem")}</Chip> : <Chip tone="pass">no problems</Chip>}
          <button type="button" className="btn" disabled={!dirty || saving} onClick={() => { setRows(baseline.current); setErrors(whole.data?.errors ?? []); setSelected(null); }}>
            <Undo2 size={14} aria-hidden="true" /> Revert
          </button>
          <button type="button" className="btn btn-primary" disabled={!writable || !dirty || saving || errors.length > 0} onClick={() => void onSave()}>
            {saving ? <><Spinner size={14} /> Saving</> : <><Save size={14} aria-hidden="true" /> Save file</>}
          </button>
        </>
      }
    >
      {!writable ? (
        <Banner tone="warn">Read-only: committed file; only *.sample.jsonl in examples/ or gitignored paths are editable.</Banner>
      ) : null}
      {reviewCount > 0 ? (
        <Banner tone="info">
          Review mode: {plural(reviewCount, "row")} came from LangSmith and still have review_required set. Approve each one to turn its observed output into the expected output; pushing is refused until none are left.
        </Banner>
      ) : null}
      {saved ? <Banner tone="success"><span className="inline-flex items-center gap-1"><CircleCheck size={14} aria-hidden="true" /> Saved.</span></Banner> : null}
      {saveError ? <Banner tone="error">{saveError}</Banner> : null}
      {errors.length ? (
        <section aria-label="Row problems" className="card card-pad mb-4">
          <h2 className="mb-2 text-[14px]">{plural(errors.length, "problem")} in this file</h2>
          <ul className="m-0 grid list-none gap-1 p-0 text-[13.5px]">
            {errors.slice(0, 12).map((e, i) => (
              <li key={i} className="flex items-start gap-2">
                <button type="button" className="btn btn-sm shrink-0" onClick={() => { setPage(Math.floor((e.line - 1) / PAGE)); setSelected(e.line - 1); }} aria-label={`Jump to line ${e.line}`}>
                  Line {e.line}
                </button>
                <span className="pre-text pt-0.5 text-fail">{e.message}</span>
              </li>
            ))}
            {errors.length > 12 ? <li className="muted">and {errors.length - 12} more</li> : null}
          </ul>
        </section>
      ) : null}

      {rows.length === 0 ? (
        <EmptyState title="This file has no rows" action={writable ? <button className="btn btn-primary" onClick={addRow}>Add the first row</button> : undefined}>
          Rows are one JSON object per line.
        </EmptyState>
      ) : (
        <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(320px,1fr)]">
          <div>
            <div className="card overflow-x-auto">
              <table className="tbl">
                <caption className="sr-only">Dataset rows, page {page + 1} of {pages}</caption>
                <thead>
                  <tr>
                    <th scope="col" className="num">Line</th>
                    <th scope="col">Id</th>
                    <th scope="col">Input</th>
                    <th scope="col">Segments</th>
                    <th scope="col">State</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((r, k) => {
                    const i = page * PAGE + k;
                    const line = i + 1;
                    const err = errorsByLine.get(line);
                    const review = r?.["review_required"] === true;
                    return (
                      <tr
                        key={i}
                        className="row-link cursor-pointer"
                        tabIndex={0}
                        aria-selected={selected === i}
                        onClick={() => setSelected(i)}
                        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSelected(i); } }}
                      >
                        <td className="num muted">{line}</td>
                        <td className="mono text-[13px]">{rowText(r?.["id"]) || <span className="muted">none</span>}</td>
                        <td className="max-w-[280px] truncate">{rowPreview(r)}</td>
                        <td className="whitespace-nowrap text-[12.5px]">
                          {["risk", "difficulty", "split", "persona"].map((k2) => (rowText(r?.[k2]) ? <Chip key={k2}>{rowText(r?.[k2])}</Chip> : null))}
                        </td>
                        <td>
                          <div className="flex flex-wrap gap-1">
                            {err ? <Chip tone="fail" title={err}>error</Chip> : null}
                            {review ? <Chip tone="warn">needs review</Chip> : null}
                            {r === null ? <Chip tone="fail">not an object</Chip> : null}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex items-center justify-between">
              <button type="button" className="btn btn-sm" onClick={addRow} disabled={!writable}><Plus size={14} aria-hidden="true" /> Add row</button>
              {pages > 1 ? (
                <div className="flex items-center gap-2 text-[13px]">
                  <button type="button" className="btn btn-sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
                  <span aria-live="polite">Page {page + 1} of {pages}</span>
                  <button type="button" className="btn btn-sm" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next</button>
                </div>
              ) : null}
            </div>
          </div>

          <aside aria-label="Row editor" className="card card-pad lg:sticky lg:top-4">
            {selected === null || (sel === null && rows[selected] !== null) ? (
              <p className="muted m-0 text-[14px]">Select a row to see and edit its JSON.</p>
            ) : (
              <RowPanel
                key={`${selected}-${rev}`}
                line={selected + 1}
                row={sel}
                writable={writable}
                error={errorsByLine.get(selected + 1) ?? null}
                jsonError={rowJsonError}
                onChange={(text) => {
                  try {
                    const v: unknown = JSON.parse(text);
                    if (typeof v !== "object" || v === null || Array.isArray(v)) throw new Error("Row must be a JSON object.");
                    setRowJsonError(null);
                    update(selected, v as Record<string, unknown>);
                  } catch (e) {
                    setRowJsonError(e instanceof Error ? e.message : "Invalid JSON");
                  }
                }}
                onApprove={(out) => { approve(selected, out); setRev((n) => n + 1); }}
                onDelete={() => { setRows(rows.filter((_, j) => j !== selected)); setSelected(null); }}
              />
            )}
          </aside>
        </div>
      )}
    </Page>
  );
}

function RowPanel({ line, row, writable, error, jsonError, onChange, onApprove, onDelete }: {
  line: number;
  row: Row;
  writable: boolean;
  error: string | null;
  jsonError: string | null;
  onChange: (text: string) => void;
  onApprove: (expectedOutput: string) => void;
  onDelete: () => void;
}) {
  const review = row?.["review_required"] === true;
  const observed = typeof row?.["observed_output"] === "string" ? row["observed_output"] : "";
  const [expected, setExpected] = useState(observed);
  // Held in state so typing in the editor never feeds back into its own value (cursor jumps).
  const [initial] = useState(() => JSON.stringify(row ?? {}, null, 2));
  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between">
        <h2 className="text-[15px]">Line {line}</h2>
        <button type="button" className="btn btn-sm btn-danger" disabled={!writable} onClick={onDelete}><Trash2 size={13} aria-hidden="true" /> Delete row</button>
      </div>
      {error ? <p role="alert" className="pre-text m-0 rounded-md bg-fail-soft px-3 py-2 text-[13px] text-fail">{error}</p> : null}
      <CodeEditor label={`JSON for line ${line}`} language="json" height={300} readOnly={!writable} value={initial} onChange={onChange} />
      {jsonError ? <p role="alert" className="m-0 text-[13px] text-fail">{jsonError}</p> : null}
      {review ? (
        <div className="grid gap-1.5 rounded-lg bg-warn-soft px-3 py-3">
          <label htmlFor="expected-output" className="text-[13px] font-medium">Expected output (starts from the observed output)</label>
          <textarea id="expected-output" className="input" rows={3} value={expected} onChange={(e) => setExpected(e.target.value)} disabled={!writable} />
          <button type="button" className="btn btn-primary btn-sm justify-self-start" disabled={!writable} onClick={() => onApprove(expected)}>Approve row</button>
        </div>
      ) : null}
    </div>
  );
}
