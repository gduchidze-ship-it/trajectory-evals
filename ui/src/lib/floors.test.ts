import { describe, expect, it } from "vitest";
import type { CandidateRecord } from "../api/types";
import { evaluateFloors, lineage } from "./floors";
import { isShare, signed, pct } from "./format";

const rec = (over: Partial<CandidateRecord> & { id: string }): CandidateRecord => ({
  parent: null,
  verdict: "accepted",
  reason: null,
  rationale: null,
  cost: null,
  val: null,
  judge_coverage: {},
  redteam: null,
  fixed_cases: [],
  regressed_cases: [],
  sign_test_p: null,
  deltas: {},
  agent_calls: 0,
  ...over,
});

describe("evaluateFloors", () => {
  const optimize = { floors: { pass_rate: 0.6, max_error_rate: 0.1, latency_p95_ms: 5000, min_judge_coverage: 0.9 } };
  it("marks each floor passed or failed from the val summary", () => {
    const c = rec({ id: "c", val: { pass_rate: 0.5, error_rate: 0.1, latency_p95_ms: 6200 }, judge_coverage: { helpful: 0.7, tone: 1 } });
    const byKey = Object.fromEntries(evaluateFloors(optimize, c).map((f) => [f.key, f]));
    expect(byKey["pass_rate"]).toMatchObject({ status: "fail", measured: 0.5 });
    expect(byKey["max_error_rate"]?.status).toBe("pass");
    expect(byKey["latency_p95_ms"]?.status).toBe("fail");
    expect(byKey["min_judge_coverage"]).toMatchObject({ status: "fail", measured: 0.7 });
  });
  it("shows a floor with no measured value as unmeasured, and skips a candidate that was never evaluated", () => {
    const f = evaluateFloors({ floors: { pass_rate: 0.5 } }, rec({ id: "c", val: { error_rate: 0 } }));
    expect(f).toEqual([expect.objectContaining({ key: "pass_rate", status: "unmeasured", measured: null })]);
    expect(evaluateFloors({ floors: { pass_rate: 0.5 } }, rec({ id: "c", val: null }))).toEqual([]);
  });
  it("returns nothing when no floors are configured", () => {
    expect(evaluateFloors({}, rec({ id: "c" }))).toEqual([]);
  });
});

describe("lineage", () => {
  it("orders children after their parent and reports depth and parent val", () => {
    const nodes = lineage([
      rec({ id: "baseline", val: { pass_rate: 0.5 } }),
      rec({ id: "c1", parent: "baseline", val: { pass_rate: 0.6 } }),
      rec({ id: "c2", parent: "c1" }),
      rec({ id: "c3", parent: "baseline" }),
    ]);
    expect(nodes.map((n) => [n.record.id, n.depth, n.parentVal])).toEqual([
      ["baseline", 0, null],
      ["c1", 1, 0.5],
      ["c2", 2, 0.6],
      ["c3", 1, 0.5],
    ]);
  });
  it("treats an unknown parent as a root and never loops", () => {
    expect(lineage([rec({ id: "a", parent: "zzz" }), rec({ id: "b", parent: "b" })]).map((n) => n.record.id)).toEqual(["a"]);
  });
});

describe("isShare", () => {
  it("recognises proportions by key, including segment keys", () => {
    for (const k of ["pass_rate", "error_rate", "pass_rate_ci_low", "pass_rate[risk=high]", "attack_succeeded", "contains_coverage", "defended", "pass_all_k"]) expect(isShare(k)).toBe(true);
    for (const k of ["latency_p95_ms", "rows", "repetitions", "contains"]) expect(isShare(k)).toBe(false);
  });
});

describe("format", () => {
  it("formats signed share deltas and nulls", () => {
    expect(signed(0.25, true)).toBe("+25.0 pp");
    expect(signed(-0.1, true)).toBe("−10.0 pp");
    expect(pct(null)).toBe("–");
  });
});
