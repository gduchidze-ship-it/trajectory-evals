import { CircleCheck, FileText, Save } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api, describeError } from "../api/client";
import { useAgent, useAgentFile, useDatasets, useResults, useSaveFile } from "../api/hooks";
import type { AgentDetail as Detail, AgentFile } from "../api/types";
import { CodeEditor } from "../components/CodeEditor";
import { ResultsTable } from "../components/ResultsTable";
import { Tab, TabList, TabPanel, Tabs } from "../components/tabs";
import { Banner, Chip, EmptyState, ErrorState, Page, Section, SkeletonRows, Spinner } from "../components/ui";
import { useDebounced } from "../lib/hooks";
import { useUnsavedGuard } from "../lib/unsaved";
import { datasetHref, launchHref } from "../lib/links";
import { dateTime, plural } from "../lib/format";
import { ApiError } from "../api/client";
import { CapabilityIcons } from "./Agents";

const TABS = ["overview", "files", "datasets", "runs"] as const;
type TabName = (typeof TABS)[number];

export function AgentDetailPage() {
  const { agentId = "" } = useParams();
  const [sp, setSp] = useSearchParams();
  const q = useAgent(agentId);
  const tab: TabName = TABS.find((t) => t === sp.get("tab")) ?? "overview";

  if (q.isPending) return <Page title="Agent"><SkeletonRows rows={6} label="Loading agent" /></Page>;
  if (q.isError) {
    const gone = q.error instanceof ApiError && q.error.status === 404;
    return (
      <Page title={gone ? "Agent not found" : "Agent"} back={<Link to="/agents">Back to agents</Link>}>
        {gone ? (
          <EmptyState title="This agent is gone">
            {q.error.message} It may have been renamed or removed. <Link to="/agents">See all agents</Link>.
          </EmptyState>
        ) : (
          <ErrorState error={q.error} onRetry={() => void q.refetch()} />
        )}
      </Page>
    );
  }
  const a = q.data;
  return (
    <Page
      title={a.name}
      back={<Link to="/agents">Agents</Link>}
      subtitle={a.description ?? "No description."}
      actions={
        <>
          <Chip tone={a.valid ? "pass" : "fail"}>{a.valid ? "Valid" : "Invalid"}</Chip>
          <CapabilityIcons caps={a.capabilities} />
          <Link className="btn btn-primary" to={launchHref({ agent: a.id })}>Run this agent</Link>
        </>
      }
    >
      {a.error ? <Banner tone="error"><Chip tone="fail">{a.error.kind}</Chip> {a.error.message}</Banner> : null}
      <Tabs value={tab} onValueChange={(v) => setSp({ tab: v }, { replace: true })}>
        <TabList aria-label="Agent sections">
          <Tab value="overview">Overview</Tab>
          <Tab value="files">Files <span className="muted">({a.files.length})</span></Tab>
          <Tab value="datasets">Datasets <span className="muted">({a.dataset_list.length})</span></Tab>
          <Tab value="runs">Runs</Tab>
        </TabList>
        <TabPanel value="overview"><Overview a={a} /></TabPanel>
        <TabPanel value="files"><FilesTab a={a} /></TabPanel>
        <TabPanel value="datasets"><DatasetsTab a={a} /></TabPanel>
        <TabPanel value="runs"><RunsTab a={a} /></TabPanel>
      </Tabs>
    </Page>
  );
}

function Json({ value, label }: { value: unknown; label: string }) {
  return (
    <details className="card card-pad">
      <summary className="cursor-pointer text-[14px] font-medium">{label}</summary>
      <pre className="pre-text mt-2 text-[12.5px]">{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

function Overview({ a }: { a: Detail }) {
  return (
    <div className="grid gap-5">
      <Section title="Endpoint">
        {a.agent ? (
          <dl className="card card-pad grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1.5 text-[14px]">
            <dt className="muted">Kind</dt><dd className="m-0">{a.agent.kind}</dd>
            {a.agent.url ? (<><dt className="muted">URL</dt><dd className="mono m-0 text-[13px]">{a.agent.url}</dd></>) : null}
            {a.agent.responder ? (<><dt className="muted">Responder</dt><dd className="mono m-0 text-[13px]">{a.agent.responder}</dd></>) : null}
            <dt className="muted">History</dt><dd className="m-0">{a.agent.history}</dd>
            <dt className="muted">Timeout</dt><dd className="m-0">{a.agent.timeout_s} s</dd>
            {a.agent.auth_token_env ? (
              <><dt className="muted">Auth token</dt><dd className="m-0"><span className="mono text-[13px]">{a.agent.auth_token_env}</span> {a.agent.auth_token_set ? <Chip tone="pass">set</Chip> : <Chip tone="warn">not set</Chip>}</dd></>
            ) : null}
          </dl>
        ) : (
          <EmptyState title="No endpoint">This agent's [agent] table did not load; see the error above.</EmptyState>
        )}
      </Section>
      <Section title="Judges">
        {a.judges.length ? (
          <div className="card overflow-x-auto">
            <table className="tbl">
              <thead><tr><th>Name</th><th>Prompt</th><th>Model</th><th>Mode</th></tr></thead>
              <tbody>
                {a.judges.map((j) => (
                  <tr key={j.name}>
                    <td className="font-medium">{j.name}</td>
                    <td className="mono text-[13px]">{j.prompt}</td>
                    <td className="mono text-[13px]">{j.model}</td>
                    <td>
                      <div className="flex gap-1.5">
                      {j.gate ? <Chip tone="accent">gate</Chip> : null}
                      {j.continuous ? <Chip>continuous</Chip> : <Chip>boolean</Chip>}
                      {j.reference_free ? <Chip>reference-free</Chip> : null}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="muted text-[14px]">No judges configured.</p>}
      </Section>
      <Section title="Personas">
        {a.personas.length ? (
          <ul className="m-0 flex list-none flex-wrap gap-2 p-0">
            {a.personas.map((p) => (
              <li key={p.name}><Chip tone={p.kind === "attack" ? "err" : undefined} title={`${p.kind}, from the ${p.source}`}>{p.name} · {p.kind}</Chip></li>
            ))}
          </ul>
        ) : <p className="muted text-[14px]">No personas: simulation and red-team are not configured.</p>}
      </Section>
      <Section title="Configuration tables">
        <div className="grid gap-2">
          <Json label="Parsed tables" value={a.tables} />
          {a.simulation ? <Json label="Simulation" value={a.simulation} /> : null}
          {a.redteam ? <Json label="Red-team" value={a.redteam} /> : null}
          {a.optimize ? <Json label="Optimize" value={a.optimize} /> : null}
          {Object.keys(a.variants).length ? <Json label="Variants" value={a.variants} /> : null}
          {a.langsmith ? <Json label="LangSmith" value={a.langsmith} /> : null}
        </div>
      </Section>
    </div>
  );
}

type Validation = { state: "idle" } | { state: "checking" } | { state: "ok" } | { state: "error"; message: string };

function FilesTab({ a }: { a: Detail }) {
  const [path, setPath] = useState<string | null>(a.files[0]?.path ?? null);
  const [dirty, setDirty] = useState(false);
  if (!a.files.length) return <EmptyState title="No editable files" icon={<FileText size={28} />}>This agent has no agent.toml or prompt files that loaded.</EmptyState>;
  const select = (p: string) => {
    if (p !== path && dirty && !window.confirm("Discard unsaved edits to this file?")) return;
    setDirty(false);
    setPath(p);
  };
  const file = a.files.find((f) => f.path === path);
  return (
    <div className="grid gap-4 md:grid-cols-[220px_1fr]">
      <nav aria-label="Agent files">
        <ul className="m-0 grid list-none gap-1 p-0">
          {a.files.map((f) => (
            <li key={f.path}>
              <button
                type="button"
                onClick={() => select(f.path)}
                aria-current={f.path === path ? "true" : undefined}
                className={`btn btn-ghost w-full justify-start ${f.path === path ? "bg-accent-soft text-accent-ink" : ""}`}
              >
                <FileText size={14} aria-hidden="true" />
                <span className="mono truncate text-[13px]">{f.path}</span>
                {f.path === path && dirty ? <span className="ml-auto text-warn" aria-label="unsaved changes">●</span> : null}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      {file ? <FileEditor key={file.path} agentId={a.id} file={file} onDirty={setDirty} /> : null}
    </div>
  );
}

function FileEditor({ agentId, file, onDirty }: { agentId: string; file: AgentFile; onDirty: (d: boolean) => void }) {
  const q = useAgentFile(agentId, file.path);
  const save = useSaveFile(agentId);
  const [draft, setDraft] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [validation, setValidation] = useState<Validation>({ state: "idle" });
  const [justSaved, setJustSaved] = useState(false);

  useEffect(() => {
    if (q.data && draft === null) {
      setDraft(q.data.text);
      setSaved(q.data.text);
    }
  }, [q.data, draft]);

  const debounced = useDebounced(draft, 500);
  const dirty = draft !== null && saved !== null && draft !== saved;
  useEffect(() => onDirty(dirty), [dirty, onDirty]);
  useUnsavedGuard(dirty);

  // Dry-run validation after 500 ms without typing; a stale response is aborted, never shown.
  useEffect(() => {
    if (debounced === null || saved === null || debounced === saved) {
      setValidation({ state: "idle" });
      return;
    }
    const ctl = new AbortController();
    setValidation({ state: "checking" });
    api
      .writeFile(agentId, file.path, debounced, true, ctl.signal)
      .then(() => setValidation({ state: "ok" }))
      .catch((e: unknown) => {
        if (!ctl.signal.aborted) setValidation({ state: "error", message: describeError(e) });
      });
    return () => ctl.abort();
  }, [debounced, saved, agentId, file.path]);

  if (q.isPending) return <SkeletonRows rows={8} label="Loading file" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;

  const settled = draft === debounced;
  const canSave = file.editable && dirty && settled && validation.state === "ok" && !save.isPending;
  const onSave = () => {
    if (draft === null) return;
    save.mutate(
      { path: file.path, text: draft },
      {
        onSuccess: () => {
          setSaved(draft);
          setJustSaved(true);
          window.setTimeout(() => setJustSaved(false), 3000);
        },
      },
    );
  };

  return (
    <div className="min-w-0">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="mono text-[13px] font-medium">{file.path}</span>
          <Chip>{file.kind}</Chip>
          {!file.editable ? <Chip tone="warn">read-only</Chip> : null}
        </div>
        <div className="flex items-center gap-3">
          <span aria-live="polite" className="text-[13px]">
            {validation.state === "checking" ? <span className="muted inline-flex items-center gap-1"><Spinner size={13} /> Validating</span> : null}
            {validation.state === "ok" ? <span className="inline-flex items-center gap-1 text-pass"><CircleCheck size={14} aria-hidden="true" /> Valid</span> : null}
            {justSaved ? <span className="text-pass">Saved</span> : null}
          </span>
          <button type="button" className="btn btn-primary" disabled={!canSave} onClick={onSave}>
            {save.isPending ? <><Spinner size={14} /> Saving</> : <><Save size={14} aria-hidden="true" /> Save</>}
          </button>
        </div>
      </div>
      <CodeEditor
        label={`Editor for ${file.path}`}
        path={`${agentId}/${file.path}`}
        value={draft ?? ""}
        onChange={(v) => { setDraft(v); setJustSaved(false); }}
        language={file.path.endsWith(".toml") ? "ini" : "markdown"}
        readOnly={!file.editable}
      />
      {validation.state === "error" ? (
        <p role="alert" className="pre-text mt-2 rounded-md bg-fail-soft px-3 py-2 text-[13.5px] text-fail">{validation.message}</p>
      ) : null}
      {save.isError ? <p role="alert" className="pre-text mt-2 rounded-md bg-fail-soft px-3 py-2 text-[13.5px] text-fail">{describeError(save.error)}</p> : null}
      <p className="muted mt-2 text-[12.5px]">Edits are validated against the agent's checks before every save; nothing is written while validation fails. Tab leaves the editor with Ctrl+M.</p>
    </div>
  );
}

function DatasetsTab({ a }: { a: Detail }) {
  const list = useDatasets(a.id);
  const [creating, setCreating] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const create = async () => {
    setCreating(true);
    setErr(null);
    try {
      await api.writeDataset(`${a.path}/datasets/golden.jsonl`, { rows: [], kind: "golden" }, { create: true });
      void list.refetch();
    } catch (e) {
      setErr(describeError(e));
    } finally {
      setCreating(false);
    }
  };
  if (list.isPending) return <SkeletonRows rows={3} label="Loading datasets" />;
  if (list.isError) return <ErrorState error={list.error} onRetry={() => void list.refetch()} />;
  if (!list.data.length) {
    return (
      <>
        {err ? <Banner tone="error">{err}</Banner> : null}
        <EmptyState
          title="No datasets yet"
          action={<button className="btn btn-primary" disabled={creating} onClick={() => void create()}>{creating ? "Creating" : "Create datasets/golden.jsonl"}</button>}
        >
          Datasets are JSONL files under <code>datasets/</code>. Creating one only works where the folder is writable (a gitignored path).
        </EmptyState>
      </>
    );
  }
  return (
    <div className="card overflow-x-auto">
      <table className="tbl">
        <thead><tr><th>Dataset</th><th>Kind</th><th className="num">Rows</th><th>Problems</th><th>Modified</th></tr></thead>
        <tbody>
          {list.data.map((d) => (
            <tr key={d.path} className="row-link">
              <td><Link to={datasetHref(d.path, a.id)} className="mono text-[13px] font-medium">{d.path.split("/").slice(-1)[0]}</Link></td>
              <td><Chip>{d.kind}</Chip></td>
              <td className="num">{d.rows}</td>
              <td>
                {d.invalid_rows ? <Chip tone="fail">{plural(d.invalid_rows, "invalid row")}</Chip> : <span className="muted">none</span>}{" "}
                {!d.writable ? <Chip title="Committed file; only *.sample.jsonl in examples/ or gitignored paths are editable">read-only</Chip> : null}
              </td>
              <td>{dateTime(d.modified_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RunsTab({ a }: { a: Detail }) {
  const q = useResults({ agent: a.name, limit: 50 });
  if (q.isPending) return <SkeletonRows rows={3} label="Loading runs" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  if (!q.data.items.length) {
    return <EmptyState title="No runs for this agent yet" action={<Link className="btn btn-primary" to={launchHref({ agent: a.id })}>Launch a run</Link>} />;
  }
  return <ResultsTable items={q.data.items} showAgent={false} />;
}
