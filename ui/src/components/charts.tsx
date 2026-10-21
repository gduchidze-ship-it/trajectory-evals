import type { ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { pct } from "../lib/format";

/** Wraps a chart with a text table alternative, so no value is available only through colour or hover. */
export function ChartFrame({ title, caption, children, table }: { title: string; caption?: string; children: ReactNode; table: ReactNode }) {
  return (
    <figure className="card card-pad m-0">
      <figcaption className="mb-2">
        <div className="text-[14px] font-semibold">{title}</div>
        {caption ? <div className="muted text-[12.5px]">{caption}</div> : null}
      </figcaption>
      {children}
      <details className="mt-2 text-[13px]">
        <summary className="muted cursor-pointer">Show as table</summary>
        <div className="mt-2 overflow-x-auto">{table}</div>
      </details>
    </figure>
  );
}

/** A proportion with its confidence interval: a dot and whisker on a 0 to 100% axis. */
export function IntervalBar({ value, low, high, label }: { value: number | null; low: number | null; high: number | null; label: string }) {
  if (value === null) return <p className="muted text-[14px]">No scored rows.</p>;
  const x = (v: number) => 8 + v * 284;
  const text = `${label}: ${pct(value, 1)}${low !== null && high !== null ? `, 95% interval ${pct(low, 1)} to ${pct(high, 1)}` : ""}`;
  return (
    <svg viewBox="0 0 300 44" role="img" aria-label={text} className="block w-full max-w-[420px]">
      <line x1={x(0)} x2={x(1)} y1={20} y2={20} stroke="var(--viz-grid)" strokeWidth={2} strokeLinecap="round" />
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={16} y2={24} stroke="var(--ink-3)" strokeWidth={1} />
          <text x={x(t)} y={40} textAnchor="middle" fontSize={10} fill="var(--ink-2)">
            {t * 100}%
          </text>
        </g>
      ))}
      {low !== null && high !== null ? (
        <>
          <line x1={x(low)} x2={x(high)} y1={20} y2={20} stroke="var(--viz-1)" strokeWidth={4} strokeLinecap="round" opacity={0.45} />
          <line x1={x(low)} x2={x(low)} y1={13} y2={27} stroke="var(--viz-1)" strokeWidth={2} />
          <line x1={x(high)} x2={x(high)} y1={13} y2={27} stroke="var(--viz-1)" strokeWidth={2} />
        </>
      ) : null}
      <circle cx={x(value)} cy={20} r={5} fill="var(--viz-1)" stroke="var(--surface)" strokeWidth={2} />
    </svg>
  );
}

export interface BarDatum {
  name: string;
  value: number | null;
  note?: string;
}

interface TipProps {
  active?: boolean;
  payload?: { payload: BarDatum }[];
}

function BarTip({ active, payload }: TipProps) {
  const d = payload?.[0]?.payload;
  if (!active || !d) return null;
  return (
    <div className="rounded-md border border-line bg-surface px-2.5 py-1.5 text-[12.5px] shadow-lg">
      <div className="font-semibold">{d.name}</div>
      <div>{d.value === null ? "not scored" : pct(d.value, 1)}</div>
      {d.note ? <div className="muted">{d.note}</div> : null}
    </div>
  );
}

/** Horizontal bars on a fixed 0..1 axis; one series, so one hue and no legend. */
export function ShareBars({ data, ariaLabel }: { data: BarDatum[]; ariaLabel: string }) {
  const rows = data.map((d) => ({ ...d, plot: d.value ?? 0 }));
  const height = 28 + rows.length * 30;
  return (
    <div role="img" aria-label={ariaLabel} style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%" initialDimension={{ width: 480, height }}>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 44, bottom: 4, left: 4 }} barCategoryGap={8}>
          <CartesianGrid horizontal={false} stroke="var(--viz-grid)" />
          <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v: number) => pct(v)} tick={{ fill: "var(--ink-2)", fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={150} tick={{ fill: "var(--ink)", fontSize: 12 }} axisLine={false} tickLine={false} />
          <Tooltip content={<BarTip />} cursor={{ fill: "var(--sunken)" }} />
          <Bar dataKey="plot" barSize={12} radius={[0, 4, 4, 0]} isAnimationActive={false} stroke="var(--surface)" strokeWidth={2}>
            {rows.map((d) => (
              <Cell key={d.name} fill={d.value === null ? "var(--viz-grid)" : "var(--viz-1)"} />
            ))}
            <LabelList dataKey="value" position="right" formatter={(v: unknown) => (typeof v === "number" ? pct(v) : "n/a")} fill="var(--ink)" fontSize={12} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function BarTable({ data, valueHeader }: { data: BarDatum[]; valueHeader: string }) {
  return (
    <table className="tbl">
      <thead>
        <tr>
          <th>Name</th>
          <th className="num">{valueHeader}</th>
        </tr>
      </thead>
      <tbody>
        {data.map((d) => (
          <tr key={d.name}>
            <td>{d.name}</td>
            <td className="num">{d.value === null ? "not scored" : pct(d.value, 1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** A thin meter for coverage-style shares. The label sits beside it, never inside. */
export function Meter({ value, warn, label }: { value: number | null; warn?: boolean; label: string }) {
  const w = value === null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={1} aria-valuenow={value ?? undefined} className="h-2 w-full min-w-[80px] rounded-full bg-sunken">
      <div className="h-full rounded-full" style={{ width: `${w}%`, background: warn ? "var(--warn)" : "var(--viz-1)" }} />
    </div>
  );
}
