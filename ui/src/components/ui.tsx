import * as Dialog from "@radix-ui/react-dialog";
import * as Tooltip from "@radix-ui/react-tooltip";
import clsx from "clsx";
import { CircleAlert, CircleCheck, CircleDashed, CircleX, Info, LoaderCircle, TriangleAlert, X } from "lucide-react";
import type { ReactNode } from "react";
import { describeError, ApiError } from "../api/client";
import type { RowStatus } from "../api/types";
import { useDocumentTitle } from "../lib/hooks";

const STATUS: Record<RowStatus, { cls: string; label: string; Icon: typeof CircleCheck; tip: string }> = {
  passed: { cls: "chip-pass", label: "Passed", Icon: CircleCheck, tip: "Every pass key scored true." },
  failed: { cls: "chip-fail", label: "Failed", Icon: CircleX, tip: "At least one pass key scored false." },
  errored: { cls: "chip-err", label: "Errored", Icon: TriangleAlert, tip: "The target raised an error for this row." },
  unscored: {
    cls: "",
    label: "Unscored",
    Icon: CircleDashed,
    tip: "No pass keys scored this row, or the run did not record its pass keys (runs before decision D3).",
  },
};

export function StatusChip({ status }: { status: RowStatus }) {
  const s = STATUS[status];
  return (
    <span className={clsx("chip", s.cls)} title={s.tip}>
      <s.Icon size={13} aria-hidden="true" />
      {s.label}
    </span>
  );
}

export function Chip({ children, tone, title }: { children: ReactNode; tone?: "pass" | "fail" | "warn" | "err" | "accent"; title?: string }) {
  return (
    <span className={clsx("chip", tone && `chip-${tone}`)} title={title}>
      {children}
    </span>
  );
}

export function Page({
  title,
  subtitle,
  actions,
  children,
  back,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  back?: ReactNode;
}) {
  useDocumentTitle(typeof title === "string" ? title : "Workbench");
  return (
    <div className="mx-auto w-full max-w-[1240px] px-6 pb-16 pt-6">
      {back ? <div className="mb-2 text-[13px]">{back}</div> : null}
      <header className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-[24px] leading-tight">{title}</h1>
          {subtitle ? <p className="muted mt-1 max-w-[70ch] text-[14px]">{subtitle}</p> : null}
        </div>
        {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
      </header>
      {children}
    </div>
  );
}

export function Section({ title, aside, children, id }: { title: ReactNode; aside?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section className="mb-6" aria-labelledby={id}>
      <div className="mb-2 flex items-end justify-between gap-3">
        <h2 id={id} className="text-[16px]">
          {title}
        </h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Spinner({ size = 16, label }: { size?: number; label?: string }) {
  return <LoaderCircle size={size} className="spin" role={label ? "status" : undefined} aria-label={label} aria-hidden={label ? undefined : true} />;
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("skeleton", className)} aria-hidden="true" />;
}

export function SkeletonRows({ rows = 5, label = "Loading" }: { rows?: number; label?: string }) {
  return (
    <div className="card card-pad grid gap-3" role="status" aria-label={label}>
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-6" />
      ))}
    </div>
  );
}

export function EmptyState({ title, children, action, icon }: { title: string; children?: ReactNode; action?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="card flex flex-col items-center gap-2 px-6 py-12 text-center">
      {icon ? <div className="text-ink-3" aria-hidden="true">{icon}</div> : null}
      <h3 className="text-[16px]">{title}</h3>
      {children ? <div className="muted max-w-[56ch] text-[14px]">{children}</div> : null}
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}

export function Banner({ tone = "info", children, action, role }: { tone?: "info" | "warn" | "error" | "success"; children: ReactNode; action?: ReactNode; role?: "alert" | "status" }) {
  const Icon = tone === "error" ? CircleAlert : tone === "warn" ? TriangleAlert : tone === "success" ? CircleCheck : Info;
  const colors = {
    info: "bg-accent-soft text-accent-ink",
    warn: "bg-warn-soft text-warn",
    error: "bg-fail-soft text-fail",
    success: "bg-pass-soft text-pass",
  }[tone];
  return (
    <div role={role ?? (tone === "error" ? "alert" : "status")} className={clsx("mb-4 flex items-start gap-3 rounded-lg px-4 py-3 text-[14px]", colors)}>
      <Icon size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1 [overflow-wrap:anywhere]">{children}</div>
      {action}
    </div>
  );
}

/** Verbatim ErrorBody message, with the kind chip and an optional retry. */
export function ErrorState({ error, onRetry, title = "Could not load this" }: { error: unknown; onRetry?: () => void; title?: string }) {
  const kind = error instanceof ApiError ? error.info.kind : null;
  return (
    <div role="alert" className="card flex flex-col items-start gap-2 border-fail px-6 py-6">
      <div className="flex items-center gap-2 text-fail">
        <CircleAlert size={18} aria-hidden="true" />
        <h3 className="text-[16px]">{title}</h3>
        {kind ? <Chip tone="fail">{kind}</Chip> : null}
      </div>
      <p className="pre-text text-[14px]">{describeError(error)}</p>
      {onRetry ? (
        <button type="button" className="btn btn-sm" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function Field({
  label,
  htmlFor,
  help,
  error,
  optional,
  children,
}: {
  label: string;
  htmlFor: string;
  help?: ReactNode;
  error?: string | undefined;
  optional?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-1">
      <label htmlFor={htmlFor} className="text-[13px] font-medium">
        {label}
        {optional ? <span className="muted font-normal"> (optional)</span> : null}
      </label>
      {help ? (
        <p id={`${htmlFor}-help`} className="muted text-[12.5px]">
          {help}
        </p>
      ) : null}
      {children}
      {error ? (
        <p id={`${htmlFor}-error`} className="text-[13px] text-fail">
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function describedBy(id: string, help: boolean, error: boolean): string | undefined {
  const ids = [help ? `${id}-help` : "", error ? `${id}-error` : ""].filter(Boolean).join(" ");
  return ids || undefined;
}

export function Tip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Tooltip.Root delayDuration={200}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          sideOffset={6}
          className="z-[60] max-w-[280px] rounded-md bg-ink px-2.5 py-1.5 text-[12.5px] text-bg shadow-lg"
        >
          {label}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function Drawer({
  open,
  onOpenChange,
  title,
  description,
  restoreFocus,
  children,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  title: string;
  description?: string;
  restoreFocus?: () => void;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content
          className="drawer"
          onCloseAutoFocus={(e) => {
            if (restoreFocus) {
              e.preventDefault();
              restoreFocus();
            }
          }}
        >
          <div className="flex items-start justify-between gap-3 border-b border-line px-5 py-4">
            <div className="min-w-0">
              <Dialog.Title className="text-[17px] font-semibold [overflow-wrap:anywhere]">{title}</Dialog.Title>
              <Dialog.Description className="muted text-[13px]">{description ?? "Details"}</Dialog.Description>
            </div>
            <Dialog.Close className="btn btn-ghost btn-sm" aria-label="Close">
              <X size={16} aria-hidden="true" />
            </Dialog.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function Modal({ open, onOpenChange, title, children }: { open: boolean; onOpenChange: (o: boolean) => void; title: string; children: ReactNode }) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content className="modal" aria-describedby={undefined}>
          <Dialog.Title className="mb-3 text-[17px] font-semibold">{title}</Dialog.Title>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export function Progress({ value, max, label }: { value: number; max: number | null; label: string }) {
  const pctDone = max && max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max ?? undefined}
      aria-valuenow={max ? value : undefined}
      className="h-2 w-full overflow-hidden rounded-full bg-sunken"
    >
      <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${pctDone}%` }} />
    </div>
  );
}

export const kindLabel: Record<string, string> = { eval: "Eval", simulate: "Simulate", redteam: "Red-team", optimize: "Optimize", sweep: "Sweep", harvest: "Harvest", score: "Score" };
