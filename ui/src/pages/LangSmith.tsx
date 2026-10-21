import { Cloud } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, describeError } from "../api/client";
import { useAgent, useAgents, useDatasets, useHealth, useStartRun } from "../api/hooks";
import type { Command } from "../api/types";
import { Tab, TabList, TabPanel, Tabs } from "../components/tabs";
import { Banner, Chip, EmptyState, ErrorState, Field, Page, SkeletonRows, Spinner } from "../components/ui";
import { COMMAND_SPECS, buildRequest, defaultValues, validateForm, type FormValues } from "../forms/runForms";
import { plural } from "../lib/format";
import { datasetHref } from "../lib/links";
import { FieldControl } from "./Launcher";

const TABS = ["harvest", "review", "push", "pull", "score"] as const;
type TabName = (typeof TABS)[number];

export function LangSmithPage() {
  const [sp, setSp] = useSearchParams();
  const tab: TabName = TABS.find((t) => t === sp.get("tab")) ?? "harvest";
  const health = useHealth();
  const enabled = !!health.data && (health.data.demo || health.data.langsmith_key);
  const reason = health.data && !enabled ? "LANGSMITH_API_KEY is not set. Export it and restart the server, or use demo mode (python -m trajectory_evals ui --demo)." : null;

  return (
    <Page title="LangSmith" subtitle="Bring production runs in, review them into a golden dataset, push it back, and score runs online.">
      {health.isPending ? <SkeletonRows rows={3} label="Checking LangSmith access" /> : null}
      {health.isError ? <ErrorState error={health.error} onRetry={() => void health.refetch()} /> : null}
      {reason ? <Banner tone="warn" role="status"><strong>LangSmith actions are disabled.</strong> {reason} The review queue still works on files already on disk.</Banner> : null}
      {health.data?.demo ? <Banner tone="info">Demo mode uses an in-memory LangSmith stand-in with 30 canned runs, a third of them errored.</Banner> : null}
      {health.data ? (
        <Tabs value={tab} onValueChange={(v) => setSp({ tab: v }, { replace: true })}>
          <TabList aria-label="LangSmith sections">
            <Tab value="harvest">Harvest</Tab>
            <Tab value="review">Review queue</Tab>
            <Tab value="push">Push</Tab>
            <Tab value="pull">Pull</Tab>
            <Tab value="score">Score</Tab>
          </TabList>
          <TabPanel value="harvest"><JobForm command="harvest" enabled={enabled} /></TabPanel>
          <TabPanel value="review"><ReviewQueue /></TabPanel>
          <TabPanel value="push"><PushForm enabled={enabled} /></TabPanel>
          <TabPanel value="pull"><PullForm enabled={enabled} /></TabPanel>
          <TabPanel value="score"><JobForm command="score" enabled={enabled} /></TabPanel>
        </Tabs>
      ) : null}
    </Page>
  );
}

function JobForm({ command, enabled }: { command: Extract<Command, "harvest" | "score">; enabled: boolean }) {
  const navigate = useNavigate();
  const agents = useAgents();
  const start = useStartRun();
  const [values, setValues] = useState<FormValues>(defaultValues(command));
  const [errors, setErrors] = useState<Record<string, string>>({});
  const agentId = typeof values["agent"] === "string" ? values["agent"] : "";
  const agent = useAgent(agentId);
  const spec = COMMAND_SPECS[command];
  const valid = (agents.data ?? []).filter((a) => a.valid);
  const set = (n: string, v: FormValues[string]) => { setValues((x) => ({ ...x, [n]: v })); setErrors((e) => ({ ...e, [n]: "" })); };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const errs = validateForm(command, values, { projectConfigured: agent.data?.capabilities.langsmith ?? false });
    setErrors(errs);
    if (Object.keys(errs).length) return;
    start.mutate(buildRequest(command, values), { onSuccess: (job) => navigate(`/runs/${job.id}`) });
  };

  if (agents.isPending) return <SkeletonRows rows={4} label="Loading agents" />;
  if (agents.isError) return <ErrorState error={agents.error} onRetry={() => void agents.refetch()} />;
  return (
    <form onSubmit={submit} noValidate>
      <fieldset disabled={!enabled} className="m-0 grid max-w-[720px] gap-4 border-0 p-0">
        <p className="muted m-0 text-[14px]">{spec.blurb} It runs as a job; you follow it live.</p>
        {start.isError ? <Banner tone="error">{describeError(start.error)}</Banner> : null}
        <div className="card card-pad grid gap-4">
          {spec.fields.map((f) => (
            <FieldControl key={f.name} f={f} values={values} set={set} error={errors[f.name] || undefined} valid={valid} detail={agent.data ?? null} include={[]} uploadsDisabled={false} uploadReason="" />
          ))}
        </div>
        {command === "score" && values["dry_run"] === true ? <p className="muted m-0 text-[13px]">Dry run is on: runs are scored and counted, but no feedback is written to LangSmith.</p> : null}
        <div><button type="submit" className="btn btn-primary" disabled={!enabled || start.isPending}>{start.isPending ? <><Spinner size={14} /> Starting</> : command === "harvest" ? "Start harvest" : "Start scoring"}</button></div>
      </fieldset>
    </form>
  );
}

function ReviewQueue() {
  const q = useDatasets();
  if (q.isPending) return <SkeletonRows rows={3} label="Loading review queue" />;
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  const files = q.data.filter((d) => d.path.startsWith("data/langsmith/") && d.review_required > 0);
  if (!files.length) {
    return (
      <EmptyState title="Nothing to review" icon={<Cloud size={28} />}>
        Harvested files with rows still marked review_required show up here. <Link to="/langsmith?tab=harvest">Harvest some runs</Link>.
      </EmptyState>
    );
  }
  return (
    <div className="card overflow-x-auto">
      <table className="tbl">
        <caption className="sr-only">Harvested files awaiting review</caption>
        <thead><tr><th>File</th><th className="num">Rows</th><th className="num">To review</th><th><span className="sr-only">Open</span></th></tr></thead>
        <tbody>
          {files.map((d) => (
            <tr key={d.path} className="row-link">
              <td className="mono text-[13px]">{d.path}</td>
              <td className="num">{d.rows}</td>
              <td className="num"><Chip tone="warn">{d.review_required}</Chip></td>
              <td className="text-right"><Link className="btn btn-sm btn-primary" to={datasetHref(d.path)}>Review {plural(d.review_required, "row")}</Link></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PushForm({ enabled }: { enabled: boolean }) {
  const agents = useAgents();
  const datasets = useDatasets();
  const [agent, setAgent] = useState("");
  const [data, setData] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const goldens = (datasets.data ?? []).filter((d) => d.kind === "golden");
  const stem = (data.split("/").pop() ?? "").replace(/\.jsonl$/, "");
  const agentName = agents.data?.find((a) => a.id === agent)?.name ?? "agent";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const r = await api.push({ agent, data, dataset: name.trim() || null });
      setDone(`Pushed ${r.path} as dataset ${r.name}.`);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  };
  if (agents.isPending || datasets.isPending) return <SkeletonRows rows={3} label="Loading" />;
  if (agents.isError) return <ErrorState error={agents.error} onRetry={() => void agents.refetch()} />;
  return (
    <form onSubmit={(e) => void submit(e)} noValidate>
      <fieldset disabled={!enabled} className="m-0 grid max-w-[720px] gap-4 border-0 p-0">
        <p className="muted m-0 text-[14px]">Upload a golden dataset to LangSmith. Rows still marked for review are refused.</p>
        {error ? <Banner tone="error">{error} {error.includes("review_required") ? <Link to="/langsmith?tab=review">Open the review queue</Link> : null}</Banner> : null}
        {done ? <Banner tone="success">{done}</Banner> : null}
        <div className="card card-pad grid gap-4">
          <Field label="Agent" htmlFor="push-agent">
            <select id="push-agent" className="input" value={agent} onChange={(e) => setAgent(e.target.value)} required>
              <option value="">Choose an agent</option>
              {agents.data.filter((a) => a.valid).map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
            </select>
          </Field>
          <Field label="Golden dataset" htmlFor="push-data">
            <select id="push-data" className="input" value={data} onChange={(e) => setData(e.target.value)}>
              <option value="">Choose a file</option>
              {goldens.map((d) => <option key={d.path} value={d.path}>{d.path}{d.review_required ? ` (${d.review_required} to review)` : ""}</option>)}
            </select>
          </Field>
          <Field label="LangSmith dataset name" htmlFor="push-name" optional help={<>Default: <code>{agentName}/{stem || "<file>"}</code>, or the agent's dataset_prefix when it sets one.</>}>
            <input id="push-name" className="input" value={name} onChange={(e) => setName(e.target.value)} autoComplete="off" />
          </Field>
        </div>
        <div><button type="submit" className="btn btn-primary" disabled={!enabled || busy || !agent || !data}>{busy ? <><Spinner size={14} /> Pushing</> : "Push dataset"}</button></div>
      </fieldset>
    </form>
  );
}

function PullForm({ enabled }: { enabled: boolean }) {
  const agents = useAgents();
  const [agent, setAgent] = useState("");
  const [dataset, setDataset] = useState("");
  const [out, setOut] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const r = await api.pull({ agent, dataset: dataset.trim(), out: out.trim() });
      setDone(r.path);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  };
  if (agents.isPending) return <SkeletonRows rows={3} label="Loading" />;
  if (agents.isError) return <ErrorState error={agents.error} onRetry={() => void agents.refetch()} />;
  return (
    <form onSubmit={(e) => void submit(e)} noValidate>
      <fieldset disabled={!enabled} className="m-0 grid max-w-[720px] gap-4 border-0 p-0">
        <p className="muted m-0 text-[14px]">Download a LangSmith dataset into a new file under data/.</p>
        {error ? <Banner tone="error">{error}</Banner> : null}
        {done ? <Banner tone="success">Wrote <Link to={datasetHref(done)}>{done}</Link>.</Banner> : null}
        <div className="card card-pad grid gap-4">
          <Field label="Agent" htmlFor="pull-agent">
            <select id="pull-agent" className="input" value={agent} onChange={(e) => setAgent(e.target.value)}>
              <option value="">Choose an agent</option>
              {agents.data.filter((a) => a.valid).map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
            </select>
          </Field>
          <Field label="LangSmith dataset name" htmlFor="pull-dataset"><input id="pull-dataset" className="input" value={dataset} onChange={(e) => setDataset(e.target.value)} autoComplete="off" /></Field>
          <Field label="Output file" htmlFor="pull-out" help="Must be new, and under data/."><input id="pull-out" className="input mono" placeholder="data/my-dataset.jsonl" value={out} onChange={(e) => setOut(e.target.value)} autoComplete="off" /></Field>
        </div>
        <div><button type="submit" className="btn btn-primary" disabled={!enabled || busy || !agent || !dataset.trim() || !out.trim()}>{busy ? <><Spinner size={14} /> Pulling</> : "Pull dataset"}</button></div>
      </fieldset>
    </form>
  );
}
