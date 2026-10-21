import clsx from "clsx";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { describeError } from "../api/client";
import { useAgent, useAgents, useHealth, useResults, useStartRun } from "../api/hooks";
import { COMMANDS, type AgentDetail, type Command } from "../api/types";
import { Banner, Chip, EmptyState, ErrorState, Field, Page, SkeletonRows, Spinner, describedBy } from "../components/ui";
import { COMMAND_SPECS, buildRequest, defaultValues, validateForm, type FieldSpec, type FormValues } from "../forms/runForms";

const isCommand = (s: string | null): s is Command => COMMANDS.some((c) => c === s);

/** Finds the field a server message is about, so it can be highlighted. */
export function fieldNamed(message: string, fields: readonly FieldSpec[]): string | null {
  const lower = message.toLowerCase();
  const hit = fields.find((f) => new RegExp(`(^|[^a-z_])${f.name}([^a-z_]|$)`).test(lower) || lower.includes(f.label.toLowerCase()));
  return hit?.name ?? null;
}

export function LauncherPage() {
  const [sp] = useSearchParams();
  const navigate = useNavigate();
  const health = useHealth();
  const agents = useAgents();
  const start = useStartRun();
  const initialCmd = sp.get("command");
  const [command, setCommand] = useState<Command>(isCommand(initialCmd) ? initialCmd : "eval");
  const [values, setValues] = useState<FormValues>(() => ({ ...defaultValues(isCommand(initialCmd) ? initialCmd : "eval"), agent: sp.get("agent") ?? "", data: sp.get("data") ?? "" }));
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [serverError, setServerError] = useState<string | null>(null);
  const [serverField, setServerField] = useState<string | null>(null);
  const formRef = useRef<HTMLFormElement>(null);

  const agentId = typeof values["agent"] === "string" ? values["agent"] : "";
  const agent = useAgent(agentId);
  const detail = agentId && agent.data ? agent.data : null;
  const optimizeRuns = useResults({ agent: detail?.name, kind: "optimize", limit: 50 });

  const spec = COMMAND_SPECS[command];
  const demo = health.data?.demo ?? false;
  const key = health.data?.langsmith_key ?? false;
  const uploadsDisabled = !health.data || demo || !key;
  const uploadReason = demo ? "Demo mode keeps every run local." : !key ? "LANGSMITH_API_KEY is not set, so runs stay local." : "";

  // `local` is forced on whenever uploading is impossible (contract C.4).
  useEffect(() => {
    if (uploadsDisabled && "local" in values && values["local"] !== true) setValues((v) => ({ ...v, local: true, upload_as_experiments: false }));
  }, [uploadsDisabled, values]);

  const chooseCommand = (c: Command) => {
    setCommand(c);
    setErrors({});
    setServerError(null);
    setValues({ ...defaultValues(c), agent: values["agent"] ?? "", data: "" });
  };
  const set = (name: string, v: FormValues[string]) => {
    setValues((x) => {
      const next = { ...x, [name]: v };
      if (name === "agent") next["data"] = "";
      return next;
    });
    setErrors((e) => {
      if (!e[name]) return e;
      const { [name]: _drop, ...rest } = e;
      return rest;
    });
    setServerError(null);
  };

  const supported = (c: Command): string | null => {
    if (!detail) return null;
    const cap = COMMAND_SPECS[c].capability;
    if (cap && !detail.capabilities[cap]) {
      return c === "harvest" || c === "score" ? "needs [langsmith].project in agent.toml, or a project below" : `${detail.name} has no ${cap} configuration`;
    }
    return null;
  };
  // Harvest and score stay launchable without a configured project: the form then requires one.
  const blocked = (c: Command): string | null => (c === "harvest" || c === "score" ? null : supported(c));

  const includeOptions = useMemo(() => {
    // Health.results_dir is root-relative POSIX, so it joins straight onto the run folder.
    const dir = health.data?.results_dir ?? "results";
    return (optimizeRuns.data?.items ?? []).filter((r) => r.complete).map((r) => ({ label: `${r.run_id} (best: ${r.best ?? "n/a"})`, path: `${dir}/${r.run_id}/best` }));
  }, [optimizeRuns.data, health.data]);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    const errs = validateForm(command, values, { projectConfigured: detail?.capabilities.langsmith ?? false });
    setErrors(errs);
    setServerError(null);
    const first = Object.keys(errs)[0];
    if (first) {
      requestAnimationFrame(() => formRef.current?.querySelector<HTMLElement>(`#f-${first}`)?.focus());
      return;
    }
    start.mutate(buildRequest(command, values), {
      onSuccess: (job) => navigate(`/runs/${job.id}`),
      onError: (err) => {
        const msg = describeError(err);
        setServerError(msg);
        setServerField(fieldNamed(msg, spec.fields));
      },
    });
  };

  if (agents.isPending) return <Page title="Launch a run"><SkeletonRows rows={5} label="Loading agents" /></Page>;
  if (agents.isError) return <Page title="Launch a run"><ErrorState error={agents.error} onRetry={() => void agents.refetch()} /></Page>;
  const valid = agents.data.filter((a) => a.valid);
  if (!valid.length) {
    return (
      <Page title="Launch a run">
        <EmptyState title="No valid agents to run">
          Fix an agent's configuration first. <Link to="/agents">See agents</Link>.
        </EmptyState>
      </Page>
    );
  }

  const main = spec.fields.filter((f) => !f.advanced);
  const advanced = spec.fields.filter((f) => f.advanced);
  const previewBody = (() => {
    try {
      return JSON.stringify(buildRequest(command, values), null, 2);
    } catch {
      return "";
    }
  })();
  const errorList = Object.entries(errors);

  return (
    <Page title="Launch a run" subtitle="Pick a command, an agent and its data. The run starts at once and you follow it live.">
      <form ref={formRef} onSubmit={onSubmit} noValidate className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="grid gap-5">
          <fieldset className="m-0 border-0 p-0">
            <legend className="eyebrow mb-2">Command</legend>
            <div role="radiogroup" aria-label="Command" className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {COMMANDS.map((c) => {
                const why = blocked(c);
                const s = COMMAND_SPECS[c];
                return (
                  <label
                    key={c}
                    className={clsx(
                      "card relative cursor-pointer px-3 py-2.5 text-[14px] has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2",
                      command === c && "border-accent bg-accent-soft",
                      why && "cursor-not-allowed opacity-55",
                    )}
                    style={{ outlineColor: "var(--focus)" }}
                    title={why ?? s.blurb}
                  >
                    <input type="radio" name="command" value={c} checked={command === c} disabled={!!why} onChange={() => chooseCommand(c)} className="sr-only" />
                    <span className="block font-semibold">{s.label}</span>
                    <span className="muted block text-[12px] leading-snug">{why ?? s.blurb}</span>
                  </label>
                );
              })}
            </div>
          </fieldset>

          {errorList.length ? (
            <div role="alert" className="rounded-lg bg-fail-soft px-4 py-3 text-[14px] text-fail">
              <strong>Fix {errorList.length === 1 ? "this field" : `these ${errorList.length} fields`} before launching</strong>
              <ul className="m-0 mt-1 list-disc pl-5">
                {errorList.map(([f, m]) => (
                  <li key={f}><a href={`#f-${f}`} onClick={(e) => { e.preventDefault(); formRef.current?.querySelector<HTMLElement>(`#f-${f}`)?.focus(); }}>{m}</a></li>
                ))}
              </ul>
            </div>
          ) : null}
          {serverError ? <Banner tone="error">{serverError}</Banner> : null}
          {spec.langsmith && uploadsDisabled && !demo ? (
            <Banner tone="warn">{spec.label} reads from LangSmith, and LANGSMITH_API_KEY is not set. Export it and restart the server to enable this command.</Banner>
          ) : null}

          <div className="card card-pad grid gap-4">
            {main.map((f) => (
              <FieldControl key={f.name} f={f} values={values} set={set} error={errors[f.name] ?? (serverField === f.name ? serverError ?? undefined : undefined)} valid={valid} detail={detail} include={includeOptions} uploadsDisabled={uploadsDisabled} uploadReason={uploadReason} />
            ))}
            {advanced.length ? (
              <details>
                <summary className="cursor-pointer text-[14px] font-medium">Advanced options</summary>
                <div className="mt-3 grid gap-4 sm:grid-cols-2">
                  {advanced.map((f) => (
                    <FieldControl key={f.name} f={f} values={values} set={set} error={errors[f.name] ?? (serverField === f.name ? serverError ?? undefined : undefined)} valid={valid} detail={detail} include={includeOptions} uploadsDisabled={uploadsDisabled} uploadReason={uploadReason} />
                  ))}
                </div>
              </details>
            ) : null}
          </div>

          <div className="flex items-center gap-3">
            <button type="submit" className="btn btn-primary" disabled={start.isPending || (spec.langsmith && uploadsDisabled && !demo)}>
              {start.isPending ? <><Spinner size={14} /> Starting</> : `Launch ${spec.label.toLowerCase()}`}
            </button>
            <span className="muted text-[13px]">Press Enter in any field to submit.</span>
          </div>
        </div>

        <aside className="card card-pad lg:sticky lg:top-4" aria-label="Request preview">
          <h2 className="eyebrow mb-2">Request</h2>
          <p className="muted mb-2 text-[12.5px]">POST /api/runs</p>
          <pre className="pre-text m-0 max-h-[420px] overflow-auto rounded-md bg-sunken p-3 text-[12px]">{previewBody}</pre>
          {demo ? <div className="mt-3"><Chip tone="accent">demo: forced local</Chip></div> : null}
        </aside>
      </form>
    </Page>
  );
}

export function FieldControl({ f, values, set, error, valid, detail, include, uploadsDisabled, uploadReason }: {
  f: FieldSpec;
  values: FormValues;
  set: (n: string, v: FormValues[string]) => void;
  error: string | undefined;
  valid: { id: string; name: string }[];
  detail: AgentDetail | null;
  include: { label: string; path: string }[];
  uploadsDisabled: boolean;
  uploadReason: string;
}) {
  const id = `f-${f.name}`;
  const val = values[f.name];
  const text = typeof val === "string" ? val : "";
  const aria = { id, "aria-invalid": error ? true : undefined, "aria-describedby": describedBy(id, !!f.help, !!error) } as const;

  if (f.kind === "bool") {
    const forced = f.upload && uploadsDisabled;
    return (
      <div className="grid gap-1">
        <label className="flex items-center gap-2 text-[14px]">
          <input {...aria} type="checkbox" checked={forced ? f.name === "local" : val === true} disabled={forced} onChange={(e) => set(f.name, e.target.checked)} />
          {f.label}
        </label>
        {forced ? <p id={`${id}-help`} className="muted text-[12.5px]">{uploadReason}</p> : f.help ? <p id={`${id}-help`} className="muted text-[12.5px]">{f.help}</p> : null}
        {error ? <p id={`${id}-error`} className="text-[13px] text-fail">{error}</p> : null}
      </div>
    );
  }
  if (f.kind === "personas") {
    const attack = (detail?.personas ?? []).filter((p) => p.kind === "attack");
    const chosen = Array.isArray(val) ? val : [];
    return (
      <fieldset className="m-0 grid gap-1 border-0 p-0" id={id}>
        <legend className="mb-1 text-[13px] font-medium">{f.label}</legend>
        {f.help ? <p className="muted text-[12.5px]">{f.help}</p> : null}
        {attack.length === 0 ? <p className="muted text-[13px]">Pick an agent with red-team personas.</p> : null}
        <div className="flex flex-wrap gap-x-5 gap-y-1.5">
          {attack.map((p) => (
            <label key={p.name} className="flex items-center gap-2 text-[14px]">
              <input type="checkbox" checked={chosen.includes(p.name)} onChange={(e) => set(f.name, e.target.checked ? [...chosen, p.name] : chosen.filter((x) => x !== p.name))} />
              {p.name}
            </label>
          ))}
        </div>
      </fieldset>
    );
  }
  if (f.kind === "agent") {
    return (
      <Field label={f.label} htmlFor={id} help={f.help} error={error}>
        <select {...aria} className="input" value={text} onChange={(e) => set(f.name, e.target.value)}>
          <option value="">Choose an agent</option>
          {valid.map((a) => <option key={a.id} value={a.id}>{a.name} ({a.id})</option>)}
        </select>
      </Field>
    );
  }
  if (f.kind === "data") {
    const options = (detail?.dataset_list ?? []).filter((d) => d.kind === f.dataKind);
    return (
      <Field label={f.label} htmlFor={id} help={f.help ?? (detail && options.length === 0 ? `This agent has no ${f.dataKind} datasets. Add one under its Datasets tab.` : undefined)} error={error}>
        <select {...aria} className="input" value={text} disabled={!detail} onChange={(e) => set(f.name, e.target.value)}>
          <option value="">{detail ? "Choose a file" : "Choose an agent first"}</option>
          {options.map((d) => <option key={d.path} value={d.path}>{d.path.split("/").slice(-1)[0]} ({d.rows} rows)</option>)}
        </select>
      </Field>
    );
  }
  if (f.kind === "include") {
    return (
      <Field label={f.label} htmlFor={id} help={f.help} error={error} optional>
        <select {...aria} className="input" value={text} onChange={(e) => set(f.name, e.target.value)}>
          <option value="">Nothing extra</option>
          {include.map((o) => <option key={o.path} value={o.path}>{o.label}</option>)}
        </select>
      </Field>
    );
  }
  return (
    <Field label={f.label} htmlFor={id} help={f.help} error={error} optional={!f.required}>
      <input {...aria} className="input" type="text" inputMode={f.kind === "int" || f.kind === "float" ? "decimal" : undefined} value={text} onChange={(e) => set(f.name, e.target.value)} autoComplete="off" />
    </Field>
  );
}
