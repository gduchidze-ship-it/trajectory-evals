import * as Tooltip from "@radix-ui/react-tooltip";
import clsx from "clsx";
import { Activity, Bot, Cloud, FlaskConical, GitCompareArrows, Keyboard, ListChecks, Monitor, Moon, Rocket, Sun } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useHealth, useRuns } from "../api/hooks";
import { useTheme } from "../lib/theme";
import { Banner, Modal } from "./ui";

const NAV = [
  { to: "/agents", label: "Agents", key: "a", Icon: Bot },
  { to: "/launch", label: "Run", key: "n", Icon: Rocket },
  { to: "/runs", label: "Jobs", key: "j", Icon: Activity },
  { to: "/results", label: "Results", key: "r", Icon: ListChecks },
  { to: "/compare", label: "Compare", key: "c", Icon: GitCompareArrows },
  { to: "/langsmith", label: "LangSmith", key: "l", Icon: Cloud },
] as const;

function typing(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  return t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName) || t.closest(".monaco-editor") !== null;
}

/** "g" then a letter jumps to a screen; "?" lists them. Inactive while typing. */
function useShortcuts(onHelp: () => void): void {
  const navigate = useNavigate();
  const armed = useRef<number | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
      if (e.key === "?") {
        e.preventDefault();
        onHelp();
        return;
      }
      if (armed.current !== null) {
        window.clearTimeout(armed.current);
        armed.current = null;
        const hit = NAV.find((n) => n.key === e.key.toLowerCase());
        if (hit) {
          e.preventDefault();
          navigate(hit.to);
        }
        return;
      }
      if (e.key === "g") armed.current = window.setTimeout(() => (armed.current = null), 1200);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate, onHelp]);
}

export function Shell(): ReactNode {
  const health = useHealth();
  const runs = useRuns();
  const { theme, cycle } = useTheme();
  const [help, setHelp] = useState(false);
  useShortcuts(() => setHelp(true));
  const active = runs.data?.filter((j) => j.state === "running" || j.state === "queued").length ?? 0;
  const ThemeIcon = theme === "light" ? Sun : theme === "dark" ? Moon : Monitor;

  return (
    <Tooltip.Provider>
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[100] focus:rounded-md focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      <div className="flex min-h-screen">
        <aside className="sticky top-0 flex h-screen w-[208px] shrink-0 flex-col border-r border-line bg-surface px-3 py-4">
          <div className="mb-5 flex items-center gap-2 px-2">
            <span className="grid h-7 w-7 place-items-center rounded-md bg-accent text-on-accent" aria-hidden="true" style={{ color: "var(--on-accent)" }}>
              <FlaskConical size={16} />
            </span>
            <span className="text-[15px] font-semibold leading-tight">Trajectory
              <br />Evals</span>
          </div>
          <nav aria-label="Main" className="grid gap-0.5">
            {NAV.map(({ to, label, key, Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  clsx(
                    "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[14px] font-medium no-underline",
                    isActive ? "bg-accent-soft text-accent-ink" : "text-ink-2 hover:bg-sunken hover:text-ink",
                  )
                }
              >
                <Icon size={16} aria-hidden="true" />
                <span className="flex-1">{label}</span>
                {to === "/runs" && active > 0 ? (
                  <span className="chip chip-accent" data-testid="active-jobs" aria-label={`${active} active ${active === 1 ? "job" : "jobs"}`}>
                    <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent" aria-hidden="true" />
                    {active}
                  </span>
                ) : (
                  <span className="kbd" aria-hidden="true">g {key}</span>
                )}
              </NavLink>
            ))}
          </nav>
          <div className="mt-auto grid gap-1">
            <button type="button" className="btn btn-ghost justify-start" onClick={cycle} aria-label={`Theme: ${theme}. Change theme`}>
              <ThemeIcon size={16} aria-hidden="true" />
              Theme: {theme}
            </button>
            <button type="button" className="btn btn-ghost justify-start" onClick={() => setHelp(true)}>
              <Keyboard size={16} aria-hidden="true" />
              Shortcuts <span className="kbd ml-auto" aria-hidden="true">?</span>
            </button>
            <p className="muted px-2 pt-1 text-[11.5px]" title={health.data?.root}>
              {health.data ? `v${health.data.version}` : health.isError ? "server unreachable" : "connecting"}
            </p>
          </div>
        </aside>
        <div className="min-w-0 flex-1">
          {health.data?.demo ? (
            <div className="px-6 pt-4">
              <Banner tone="info" role="status">
                <strong>Demo mode:</strong> scripted models, nothing leaves this machine.
              </Banner>
            </div>
          ) : null}
          {health.isError ? (
            <div className="px-6 pt-4">
              <Banner tone="error" action={<button className="btn btn-sm" onClick={() => void health.refetch()}>Retry</button>}>
                Cannot reach the API. Start it with <code>python -m trajectory_evals ui</code>.
              </Banner>
            </div>
          ) : null}
          <main id="main" tabIndex={-1} className="outline-none">
            <Outlet />
          </main>
        </div>
      </div>
      <Modal open={help} onOpenChange={setHelp} title="Keyboard shortcuts">
        <dl className="grid grid-cols-[auto_1fr] items-center gap-x-4 gap-y-2 text-[14px]">
          {NAV.map((n) => (
            <div key={n.to} className="contents">
              <dt>
                <span className="kbd">g</span> <span className="kbd">{n.key}</span>
              </dt>
              <dd className="m-0">Go to {n.label}</dd>
            </div>
          ))}
          <dt>
            <span className="kbd">/</span>
          </dt>
          <dd className="m-0">Focus the search box on a list</dd>
          <dt>
            <span className="kbd">?</span>
          </dt>
          <dd className="m-0">This list</dd>
          <dt>
            <span className="kbd">Esc</span>
          </dt>
          <dd className="m-0">Close a drawer or dialog</dd>
        </dl>
        <p className="muted mt-4 text-[13px]">Inside a code editor, press Ctrl+M (Control, not Command) to let Tab leave the editor.</p>
        <div className="mt-4 text-right">
          <button type="button" className="btn btn-primary" onClick={() => setHelp(false)}>Done</button>
        </div>
      </Modal>
    </Tooltip.Provider>
  );
}
