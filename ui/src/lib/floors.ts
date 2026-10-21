import type { CandidateRecord } from "../api/types";

export interface FloorCheck {
  key: "pass_rate" | "max_error_rate" | "latency_p95_ms" | "min_judge_coverage";
  label: string;
  rule: string;
  threshold: number;
  measured: number | null;
  /** "unmeasured" is neutral: the candidate has no value for this floor. */
  status: "pass" | "fail" | "unmeasured";
}

const judge = (measured: number | null, ok: boolean): FloorCheck["status"] => (measured === null ? "unmeasured" : ok ? "pass" : "fail");

const asNumber = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** The [optimize].floors table as dumped in report.optimize; mirrors optimize._evaluation.floor_failures. */
export function evaluateFloors(optimize: Record<string, unknown>, c: CandidateRecord): FloorCheck[] {
  if (c.val === null) return [];
  const raw = optimize["floors"];
  if (typeof raw !== "object" || raw === null) return [];
  const f = raw as Record<string, unknown>;
  const out: FloorCheck[] = [];
  const val = c.val ?? {};
  const pr = asNumber(f["pass_rate"]);
  if (pr !== null) {
    const m = val["pass_rate"] ?? null;
    out.push({ key: "pass_rate", label: "Pass rate", rule: "at least", threshold: pr, measured: m, status: judge(m, m !== null && m >= pr) });
  }
  const er = asNumber(f["max_error_rate"]);
  if (er !== null) {
    const m = val["error_rate"] ?? null;
    out.push({ key: "max_error_rate", label: "Error rate", rule: "at most", threshold: er, measured: m, status: judge(m, m !== null && m <= er) });
  }
  const lat = asNumber(f["latency_p95_ms"]);
  if (lat !== null) {
    const m = val["latency_p95_ms"] ?? null;
    out.push({ key: "latency_p95_ms", label: "Latency p95 (ms)", rule: "at most", threshold: lat, measured: m, status: judge(m, m !== null && m <= lat) });
  }
  const cov = asNumber(f["min_judge_coverage"]);
  if (cov !== null) {
    const entries = Object.entries(c.judge_coverage).filter(([, v]) => v !== null) as [string, number][];
    const worst = entries.length ? Math.min(...entries.map(([, v]) => v)) : null;
    out.push({ key: "min_judge_coverage", label: "Judge coverage (worst)", rule: "at least", threshold: cov, measured: worst, status: judge(worst, worst !== null && worst >= cov) });
  }
  return out;
}

export interface LineageNode {
  record: CandidateRecord;
  depth: number;
  parentVal: number | null;
}

/** Depth-first order from each root, so children follow their parent. Unknown parents become roots. */
export function lineage(records: readonly CandidateRecord[]): LineageNode[] {
  const ids = new Set(records.map((r) => r.id));
  const kids = new Map<string | null, CandidateRecord[]>();
  for (const r of records) {
    const key = r.parent !== null && ids.has(r.parent) ? r.parent : null;
    kids.set(key, [...(kids.get(key) ?? []), r]);
  }
  const byId = new Map(records.map((r) => [r.id, r]));
  const out: LineageNode[] = [];
  const seen = new Set<string>();
  const walk = (parent: string | null, depth: number) => {
    for (const r of kids.get(parent) ?? []) {
      if (seen.has(r.id)) continue;
      seen.add(r.id);
      const p = r.parent ? byId.get(r.parent) : undefined;
      out.push({ record: r, depth, parentVal: p?.val?.["pass_rate"] ?? null });
      walk(r.id, depth + 1);
    }
  };
  walk(null, 0);
  return out;
}
