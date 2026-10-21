import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleCheck, Plus, Save, Trash2, Undo2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { ApiError, api, describeError } from "../api/client";
import type { DatasetKind, RowError } from "../api/types";
import { CodeEditor } from "../components/CodeEditor";
import { Banner, Chip, EmptyState, ErrorState, Page, SkeletonRows, Spinner } from "../components/ui";
import { basename, plural } from "../lib/format";
import { useDebounced } from "../lib/hooks";
import { agentHref } from "../lib/links";
import { useUnsavedGuard } from "../lib/unsaved";

type Row = Record<string, unknown> | null;

/** One row of the file. `line` is its physical 1-based file line when loaded, null for a row added here. */
export interface Entry {
  line: number | null;
  data: Row;
}
export interface IndexedError {
  index: number;
  message: string;
}

const PAGE = 100;
const FETCH = 500;

interface Loaded {
  kind: DatasetKind;
  entries: Entry[];
  errors: IndexedError[];
  modifiedAt: string;
  writable: boolean;
}

/** Load-time errors carry physical file lines; map them to rows through each row's own `line`. */
export function errorsFromLines(entries: readonly Entry[], errors: readonly RowError[]): IndexedError[] {
  const out: IndexedError[] = [];
  for (const e of errors) {
    const index = entries.findIndex((x) => x.line === e.line);
    if (index !== -1) out.push({ index, message: e.message });
  }
  return out;
}

/** Validation and dry-run errors are 1-based ordinals of the posted rows. */
export const errorsFromOrdinals = (errors: readonly RowError[]): IndexedError[] => errors.map((e) => ({ index: e.line - 1, message: e.message }));

async function loadWhole(path: string): Promise<Loaded> {
  const before = (await api.datasets()).find((d) => d.path === path);
  const entries: Entry[] = [];
  let first: Awaited<ReturnType<typeof api.dataset>> | null = null;
  for (let offset = 0; ; ) {
    const p = await api.dataset(path, { offset, limit: FETCH });
    first ??= p;
    if (p.total !== first.total) throw new Error("The file changed while it was loading. Reload to try again.");
    entries.push(...p.items.map((i): Entry => ({ line: i.line, data: i.data })));
    offset += p.items.length;
    if (entries.length >= first.total || p.items.length === 0) break;
  }
  if (entries.length !== first.total) throw new Error("The server returned fewer rows than it reported. Reload to try again.");
  const after = (await api.datasets()).find((d) => d.path === path);
  if (before && after && before.modified_at !== after.modified_at) throw new Error("The file changed while it was loading. Reload to try again.");
  return {
    kind: first.kind,
    entries,
    errors: errorsFromLines(entries, first.errors),
    modifiedAt: before?.modified_at ?? "",
    writable: before?.writable ?? false,
  };
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

const sameRows = (a: readonly Entry[], b: readonly Entry[]): boolean => JSON.stringify(a.map((e) => e.data)) === JSON.stringify(b.map((e) => e.data));

export function DatasetEditorPage() {
  // React Router already decodes splat params; decoding again would corrupt a name containing "%".
  const path = useParams()["*"] ?? "";
  const [sp] = useSearchParams();
  const agent = sp.get("agent");
  const qc = useQueryClient();
  const whole = useQuery({ queryKey: ["datasets", path, "whole"], queryFn: () => loadWhole(path), staleTime: Infinity, gcTime: 0 });

  const [rows, setRows] = useState<Entry[] | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [page, setPage] = useState(0);
  const [errors, setErrors] = useState<IndexedError[]>([]);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [conflict, setConflict] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [rowJsonError, setRowJsonError] = useState<string | null>(null);
  const [rev, setRev] = useState(0);
  const baseline = useRef<Entry[]>([]);
  const modifiedAt = useRef("");

  useEffect(() => {
    if (whole.data) {
      setRows(whole.data.entries);
      baseline.current = whole.data.entries;
      modifiedAt.current = whole.data.modifiedAt;
      setErrors(whole.data.errors);
    }
  }, [whole.data]);

  useEffect(() => setRowJsonError(null), [selected]);

  const kind = whole.data?.kind ?? "golden";
  const writable = whole.data?.writable ?? false;
  const dirty = rows !== null && !sameRows(rows, baseline.current);
  useUnsavedGuard(dirty);
  const debouncedRows = useDebounced(rows, 400);

  useEffect(() => {
    if (!debouncedRows || !dirty || debouncedRows.some((r) => r.data === null)) return;
    const ctl = new AbortController();
    api
      .validateRows({ rows: debouncedRows.map((r) => r.data as Record<string, unknown>), kind }, ctl.signal)
      .then((r) => setErrors(errorsFromOrdinals(r.errors)))
      .catch(() => undefined);
    return () => ctl.abort();
  }, [debouncedRows, dirty, kind]);

  const update = (i: number, next: Row) => setRows((r) => (r ? r.map((x, j) => (j === i ? { ...x, data: next } : x)) : r));
  const errorByIndex = useMemo(() => new Map(errors.map((e) => [e.index, e.message])), [errors]);
  const lineLabel = (i: number): string => (rows?.[i]?.line ?? i + 1).toString();

  if (whole.isPending) return <Page title={basename(path)}><SkeletonRows rows={8} label="Loading dataset" /></Page>;
  if (whole.isError || rows === null) {
    return (
      <Page title={basename(path)} back={<Link to="/agents">Agents</Link>}>
        <ErrorState error={whole.error} onRetry={() => void whole.refetch()} title="Could not load this dataset" />
      </Page>
    );
  }

  const reviewCount = rows.filter((r) => r.data?.["review_required"] === true).length;
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const shown = rows.slice(page * PAGE, page * PAGE + PAGE);
  const sel = selected !== null ? rows[selected]?.data ?? null : null;
  const hasSel = selected !== null && rows[selected] !== undefined;

  const reload = async () => {
    setConflict(false);
    setSaveError(null);
    setSelected(null);
    await whole.refetch();
  };

  const onSave = async () => {
    if (rows.some((r) => r.data === null)) {
      setSaveError("A line is not a JSON object. Fix it in the side panel, or delete the row, before saving.");
      return;
    }
    setSaving(true);
    setSaveError(null);
    setConflict(false);
    try {
      await api.writeDataset(path, { rows: rows.map((r) => r.data as Record<string, unknown>), kind, expected_modified_at: modifiedAt.current || null });
      setSaved(true);
      window.setTimeout(() => setSaved(false), 3000);
      // Refetching the file also re-reads physical lines, errors and the modification stamp.
      await qc.invalidateQueries({ queryKey: ["datasets"] });
    } catch (e) {
      if (e instanceof ApiError && e.info.kind === "conflict") setConflict(true);
      else setSaveError(describeError(e));
    } finally {
      setSaving(false);
    }
  };

  const approve = (i: number, expectedOutput?: string) => {
    const r = rows[i]?.data;
    if (!r) return;
    const observed = typeof r["observed_output"] === "string" ? r["observed_output"] : "";
    const expected = { ...((typeof r["expected"] === "object" && r["expected"] !== null ? r["expected"] : {}) as Record<string, unknown>), output: expectedOutput ?? observed };
    update(i, { ...r, expected, review_required: false });
  };

  const addRow = () => {
    const n = rows.length + 1;
    const data = kind === "scenarios" ? { id: `new-${n}`, persona: "", goal: "" } : { id: `new-${n}`, input: "", expected: {} };
    setRows([...rows, { line: null, data }]);
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
          <button type="button" className="btn btn-primary" disabled={!writable || !dirty || saving || errors.length > 0 || rowJsonError !== null} onClick={() => void onSave()}>
            {saving ? <><Spinner size={14} /> Saving</> : <><Save size={14} aria-hidden="true" /> Save file</>}
          </button>
        </>
      }
    >
      {!writable ? (
        <Banner tone="warn">Read-only: committed file; only *.sample.jsonl in examples/ or gitignored paths are editable.</Banner>
      ) : null}
      {conflict ? (
        <Banner tone="error" action={<button className="btn btn-sm" onClick={() => void reload()}>Reload</button>}>
          File changed on disk, reload. Nothing was written; your edits stay here until you reload.
        </Banner>
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
                <button type="button" className="btn btn-sm shrink-0" onClick={() => { setPage(Math.floor(e.index / PAGE)); setSelected(e.index); }} aria-label={`Jump to line ${lineLabel(e.index)}`}>
                  Line {lineLabel(e.index)}
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
                  {shown.map((entry, k) => {
                    const i = page * PAGE + k;
                    const r = entry.data;
                    const err = errorByIndex.get(i);
                    const review = r?.["review_required"] === true;
                    return (
                      <tr
                        key={i}
                        className="row-link cursor-pointer"
                        tabIndex={0}
                        data-selected={selected === i}
                        onClick={() => setSelected(i)}
                        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSelected(i); } }}
                      >
                        <td className="num muted">{entry.line ?? "new"}</td>
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
            {selected === null || !hasSel ? (
              <p className="muted m-0 text-[14px]">Select a row to see and edit its JSON.</p>
            ) : (
              <RowPanel
                key={`${selected}-${rev}`}
                line={lineLabel(selected)}
                row={sel}
                writable={writable}
                error={errorByIndex.get(selected) ?? null}
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
  line: string;
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
