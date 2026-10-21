import { Bot, CircleCheck, CircleX, Cloud, FlaskConical, Search, ShieldAlert, Sparkles, Swords } from "lucide-react";
import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, describeError } from "../api/client";
import { useAgents } from "../api/hooks";
import type { AgentSummary, CheckReport } from "../api/types";
import { Banner, Chip, EmptyState, ErrorState, Page, Spinner, SkeletonRows, Tip } from "../components/ui";
import { useSlashFocus } from "../lib/hooks";

export function CapabilityIcons({ caps }: { caps: AgentSummary["capabilities"] }) {
  const items = [
    { on: caps.simulate, label: "Simulate", Icon: FlaskConical },
    { on: caps.redteam, label: "Red-team", Icon: Swords },
    { on: caps.optimize, label: "Optimize", Icon: Sparkles },
    { on: caps.langsmith, label: "LangSmith", Icon: Cloud },
  ];
  return (
    <ul className="m-0 flex list-none gap-1.5 p-0" aria-label="Capabilities">
      {items.map(({ on, label, Icon }) => (
        <li key={label} title={`${label}: ${on ? "available" : "not configured"}`} className={on ? "text-accent-ink" : "text-ink-3 opacity-40"}>
          <Icon size={16} aria-hidden="true" />
          <span className="sr-only">{`${label}: ${on ? "available" : "not configured"}`}</span>
        </li>
      ))}
    </ul>
  );
}

export function CheckSteps({ report }: { report: CheckReport }) {
  return (
    <ol className="m-0 grid list-none gap-1 p-0 text-[13.5px]" aria-label="Check steps">
      {report.steps.map((s) => (
        <li key={s.step} className="flex items-start gap-2">
          {s.ok ? <CircleCheck size={15} className="mt-0.5 text-pass" aria-label="passed" /> : <CircleX size={15} className="mt-0.5 text-fail" aria-label="failed" />}
          <div className="min-w-0">
            <span className="font-medium">{s.step}</span>
            {s.datasets.length ? <span className="muted"> · {s.datasets.map((d) => `${d.path} (${d.cases})`).join(", ")}</span> : null}
            {s.reply ? <span className="muted"> · reply received</span> : null}
            {s.error ? (
              <div className="pre-text text-fail">
                <Chip tone="fail">{s.error.kind}</Chip> {s.error.message}
              </div>
            ) : null}
          </div>
        </li>
      ))}
    </ol>
  );
}

export function AgentsPage() {
  const agents = useAgents();
  const [ping, setPing] = useState(false);
  const [filter, setFilter] = useState("");
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [reports, setReports] = useState<Record<string, CheckReport | string>>({});
  const search = useRef<HTMLInputElement>(null);
  useSlashFocus(search);

  const check = async (id: string) => {
    setPending((p) => new Set(p).add(id));
    try {
      const r = await api.check(id, ping);
      setReports((x) => ({ ...x, [id]: r }));
    } catch (e) {
      setReports((x) => ({ ...x, [id]: describeError(e) }));
    } finally {
      setPending((p) => {
        const n = new Set(p);
        n.delete(id);
        return n;
      });
      void agents.refetch();
    }
  };

  const shown = agents.data?.filter((a) => `${a.name} ${a.id} ${a.description ?? ""}`.toLowerCase().includes(filter.toLowerCase())) ?? [];

  return (
    <Page
      title="Agents"
      subtitle="Every folder with an agent.toml. Validation is offline; Check also starts the agent and fetches its card."
      actions={
        <label className="flex items-center gap-2 text-[13px]">
          <input type="checkbox" checked={ping} onChange={(e) => setPing(e.target.checked)} />
          Ping on check <span className="muted">(costs one LLM call)</span>
        </label>
      }
    >
      <div className="relative mb-4 max-w-[360px]">
        <Search size={15} className="pointer-events-none absolute left-3 top-[10px] text-ink-3" aria-hidden="true" />
        <input ref={search} className="input pl-9" type="search" placeholder="Filter agents" aria-label="Filter agents" value={filter} onChange={(e) => setFilter(e.target.value)} />
      </div>
      {agents.isPending ? (
        <SkeletonRows rows={4} label="Loading agents" />
      ) : agents.isError ? (
        <ErrorState error={agents.error} onRetry={() => void agents.refetch()} title="Could not list agents" />
      ) : agents.data.length === 0 ? (
        <EmptyState title="No agents yet" icon={<Bot size={28} />}>
          Create <code>agents/&lt;name&gt;/agent.toml</code>, or start from the ones in <code>examples/</code>.
        </EmptyState>
      ) : shown.length === 0 ? (
        <EmptyState title="No agent matches this filter" action={<button className="btn" onClick={() => setFilter("")}>Clear filter</button>} />
      ) : (
        <div className="card overflow-x-auto">
          <table className="tbl">
            <caption className="sr-only">Agents</caption>
            <thead>
              <tr>
                <th scope="col">Agent</th>
                <th scope="col">Health</th>
                <th scope="col" className="num">Datasets</th>
                <th scope="col">Can run</th>
                <th scope="col"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {shown.map((a) => {
                const rep = reports[a.id];
                return (
                  <tr key={a.id} className="row-link align-top">
                    <td className="max-w-[420px]">
                      <div className="flex flex-wrap items-center gap-2">
                        <Link to={`/agents/${encodeURIComponent(a.id)}`} className="text-[15px] font-semibold">{a.name}</Link>
                        <Chip>{a.source}</Chip>
                        {a.endpoint ? <Chip>{a.endpoint}</Chip> : null}
                      </div>
                      <p className="muted m-0 mt-0.5 line-clamp-2 text-[13px]">{a.description ?? "No description."}</p>
                      <div className="muted mono mt-0.5 text-[12px]">{a.id}</div>
                      {rep ? (
                        <div className="mt-2 rounded-md bg-sunken px-3 py-2">
                          {typeof rep === "string" ? <p className="pre-text m-0 text-fail">{rep}</p> : <CheckSteps report={rep} />}
                        </div>
                      ) : null}
                    </td>
                    <td className="max-w-[320px]">
                      {a.valid ? (
                        <Chip tone="pass"><CircleCheck size={13} aria-hidden="true" /> Valid</Chip>
                      ) : (
                        <div>
                          <Chip tone="fail"><ShieldAlert size={13} aria-hidden="true" /> Invalid</Chip> {a.error ? <Chip>{a.error.kind}</Chip> : null}
                          {a.error ? <p className="pre-text m-0 mt-1 text-[13px] text-fail">{a.error.message}</p> : null}
                        </div>
                      )}
                      {rep && typeof rep !== "string" ? (
                        <div className="mt-1.5">
                          <Chip tone={rep.ok ? "pass" : "fail"}>Check {rep.ok ? "passed" : "failed"}</Chip>
                        </div>
                      ) : null}
                    </td>
                    <td className="num">{a.datasets}</td>
                    <td><CapabilityIcons caps={a.capabilities} /></td>
                    <td className="text-right">
                      <Tip label={ping ? "Runs every check, including one live ping" : "Runs every check, including starting the agent and fetching its card"}>
                        <button type="button" className="btn btn-sm" disabled={pending.has(a.id)} onClick={() => void check(a.id)} aria-label={`Check ${a.name}`}>
                          {pending.has(a.id) ? <><Spinner size={13} /> Checking</> : "Check"}
                        </button>
                      </Tip>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {pending.size ? <Banner tone="info">Checks can take up to the agent's timeout.</Banner> : null}
    </Page>
  );
}
