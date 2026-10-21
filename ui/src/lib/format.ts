export const dash = "–";

export function pct(v: number | null | undefined, digits = 0): string {
  return v === null || v === undefined || Number.isNaN(v) ? dash : `${(v * 100).toFixed(digits)}%`;
}

export function num(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined || Number.isNaN(v) ? dash : v.toFixed(digits).replace(/\.?0+$/, "") || "0";
}

export function ms(v: number | null | undefined): string {
  if (v === null || v === undefined) return dash;
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(v < 10 ? 1 : 0)} ms`;
}

/** Signed change; `share` formats as percentage points. */
export function signed(v: number | null | undefined, share = false): string {
  if (v === null || v === undefined) return dash;
  const body = share ? `${Math.abs(v * 100).toFixed(1)} pp` : num(Math.abs(v), 3);
  return v === 0 ? body : `${v > 0 ? "+" : "−"}${body}`;
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return dash;
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? dash
    : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function relTime(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return dash;
  const s = Math.round((now - new Date(iso).getTime()) / 1000);
  if (Number.isNaN(s)) return dash;
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export const shortRun = (id: string): string => {
  const m = /^(\d{8})_(\d{6})_(.*)_([0-9a-f]{6})$/.exec(id);
  return m ? `${m[3]} · ${m[4]}` : id;
};

export function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

export const plural = (n: number, one: string, many = `${one}s`): string => `${n} ${n === 1 ? one : many}`;

/** Whether a summary key holds a proportion, to be shown as a percentage. Decided by name, never by value. */
export const isShare = (key: string): boolean =>
  /(^|_)rate(_|$)/.test(key) || key.endsWith("succeeded") || key.endsWith("_coverage") || key === "defended" || key === "pass_all_k" || key === "pass_any_k";
