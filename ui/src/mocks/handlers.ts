import { http, HttpResponse, delay } from "msw";
import type { DatasetKind, DatasetWrite, ErrorKind, Page, RowDetail, RowSummary, RunRequest, ValidationReport } from "../api/types";
import {
  AGENTS,
  DATASETS,
  EVAL_DETAILS,
  PROMPTS,
  RESULTS,
  agentDetail,
  agentFiles,
  comparison,
  detailRow,
  goldenRows,
  mockState,
  scenarioRows,
  seedsFor,
  summaryRowOf,
} from "./fixtures";
import { cancelJob, ensureFixtureJob, isTerminalJob, jobs, startJob } from "./jobs";

const err = (status: number, kind: ErrorKind, message: string, input_error = false) =>
  HttpResponse.json({ error: { kind, message, input_error } }, { status });

const datasetState = new Map<string, Record<string, unknown>[]>();
const fileState = new Map<string, string>();
const createdDatasets = new Set<string>();

export function resetMockState(): void {
  datasetState.clear();
  fileState.clear();
  createdDatasets.clear();
}

const summaryFor = (path: string) => DATASETS.find((d) => d.path === path);

/** The support golden file has a blank physical line 4, so rows after the third sit one line lower than their ordinal. */
const physicalLine = (path: string, index: number): number => index + 1 + (path.includes("support_agent/datasets/golden") && index >= 3 ? 1 : 0);

function rowsOf(path: string): Record<string, unknown>[] | null {
  const have = datasetState.get(path);
  if (have) return have;
  const s = summaryFor(path);
  if (!s) return null;
  const rows = s.kind === "scenarios" ? scenarioRows() : goldenRows(path);
  if (path.includes("support_agent/datasets/golden")) delete rows[6]?.["input"];
  datasetState.set(path, rows);
  return rows;
}

function validate(rows: Record<string, unknown>[], kind: DatasetKind): ValidationReport {
  const errors: { line: number; message: string }[] = [];
  const seen = new Set<string>();
  rows.forEach((r, i) => {
    const line = i + 1;
    const id = typeof r["id"] === "string" ? r["id"] : null;
    if (!id) errors.push({ line, message: "id: field required" });
    else if (seen.has(id)) errors.push({ line, message: "id: duplicate of an earlier row" });
    else seen.add(id);
    if (kind === "golden" && !("input" in r) && !("messages" in r)) errors.push({ line, message: "input: field required (or messages)" });
    if (kind === "scenarios" && !("goal" in r)) errors.push({ line, message: "goal: field required" });
  });
  return { ok: errors.length === 0, rows: rows.length, errors };
}

const num = (v: string | null, d: number) => (v !== null && /^\d+$/.test(v) ? Number(v) : d);

function rowMatches(r: RowSummary, p: URLSearchParams): boolean {
  const status = p.get("status");
  const q = p.get("q");
  const seg = p.get("segment");
  if (status && r.status !== status) return false;
  if (q && !r.case_id.toLowerCase().includes(q.toLowerCase())) return false;
  if (seg) {
    const [k, v] = seg.split("=");
    const val = k === "persona" ? r.persona : r.metadata[k ?? ""];
    if (String(val) !== v) return false;
  }
  return true;
}

function rowsHandler(runId: string, evalId: string | undefined, url: URL) {
  const seeds = seedsFor(runId, evalId);
  if (!seeds) return err(404, "not_found", "run not found");
  const all = seeds.map((s, i) => summaryRowOf(s, i)).filter((r) => rowMatches(r, url.searchParams));
  const offset = num(url.searchParams.get("offset"), 0);
  const limit = num(url.searchParams.get("limit"), 50);
  const page: Page<RowSummary> = { items: all.slice(offset, offset + limit), total: all.length, offset, limit };
  return HttpResponse.json(page);
}

function rowHandler(runId: string, evalId: string | undefined, index: string) {
  const seeds = seedsFor(runId, evalId);
  const s = seeds?.[Number(index)];
  if (!s) return err(404, "not_found", "row not found");
  const d: RowDetail = detailRow(s, Number(index));
  return HttpResponse.json(d);
}

export const handlers = [
  http.get("/api/health", () => HttpResponse.json(mockState.health)),

  http.get("/api/agents", async () => {
    await delay(150);
    return HttpResponse.json(AGENTS);
  }),
  http.get("/api/agents/:id", ({ params }) => {
    const a = AGENTS.find((x) => x.id === params["id"]);
    return a ? HttpResponse.json(agentDetail(a)) : err(404, "not_found", `unknown agent '${String(params["id"])}'`);
  }),
  http.post("/api/agents/:id/check", async ({ params }) => {
    const a = AGENTS.find((x) => x.id === params["id"]);
    if (!a) return err(404, "not_found", "unknown agent");
    await delay(500);
    const ok = (step: "config" | "graders" | "datasets" | "agent" | "tables" | "card") => ({ step, ok: true, error: null, datasets: [], reply: null });
    if (a.id === "examples.support_agent") {
      return HttpResponse.json({
        agent_id: a.id,
        ok: true,
        steps: [
          ok("config"),
          ok("graders"),
          { ...ok("datasets"), datasets: [{ path: "datasets/golden.sample.jsonl", cases: 12 }, { path: "datasets/scenarios.sample.jsonl", cases: 4 }] },
          ok("agent"),
          ok("tables"),
          ok("card"),
        ],
      });
    }
    return HttpResponse.json({
      agent_id: a.id,
      ok: false,
      steps: [ok("config"), ok("graders"), { ...ok("datasets"), datasets: [{ path: "datasets/golden.sample.jsonl", cases: 12 }] }, { step: "agent", ok: false, datasets: [], reply: null, error: a.error ?? { kind: "preflight", message: "cannot start agent 'support_agent': Missing credentials. Set OPENAI_API_KEY.", input_error: true } }],
    });
  }),
  http.get("/api/agents/:id/files/*", ({ params }) => {
    const path = [params["0"]].flat().join("/");
    const text = fileState.get(`${String(params["id"])}/${path}`) ?? agentFiles(String(params["id"]))[path];
    return text === undefined ? err(404, "not_found", "no such file in the agent folder") : HttpResponse.json({ path, text });
  }),
  http.put("/api/agents/:id/files/*", async ({ params, request }) => {
    const path = [params["0"]].flat().join("/");
    const id = String(params["id"]);
    const dry = new URL(request.url).searchParams.get("dry_run") === "true";
    const body = (await request.json()) as { text: string };
    if (!(path in agentFiles(id))) return err(404, "not_found", "no such file in the agent folder");
    if (path === "agent.toml" && /BROKEN/.test(body.text)) return err(422, "agent_config", "agent.toml: [agent] unknown key 'BROKEN'", true);
    if (path === "agent.toml" && !/^name\s*=/m.test(body.text)) return err(422, "agent_config", "agent.toml: name is required", true);
    if (path.endsWith(".md") && !body.text.trim()) return err(422, "agent_config", `${path}: a prompt cannot be empty`, true);
    if (!dry) fileState.set(`${id}/${path}`, body.text);
    return HttpResponse.json({ path, text: body.text });
  }),

  http.get("/api/datasets", ({ request }) => {
    const agent = new URL(request.url).searchParams.get("agent");
    if (agent && !AGENTS.some((a) => a.id === agent)) return err(404, "not_found", "unknown agent");
    return HttpResponse.json(agent ? DATASETS.filter((d) => d.agent_id === agent) : DATASETS);
  }),
  http.post("/api/datasets/validate", async ({ request }) => {
    const b = (await request.json()) as DatasetWrite;
    if (!b.kind) return err(422, "validation", "body.kind: field required");
    return HttpResponse.json(validate(b.rows, b.kind));
  }),
  http.get("/api/datasets/*", ({ params, request }) => {
    const path = [params["0"]].flat().join("/");
    const s = summaryFor(path);
    const rows = rowsOf(path);
    if (!s || !rows) return err(404, "not_found", "no such dataset");
    const p = new URL(request.url).searchParams;
    const offset = num(p.get("offset"), 0);
    const limit = num(p.get("limit"), 100);
    const rep = validate(rows, s.kind);
    const bad = new Map(rep.errors.map((e) => [e.line, e.message]));
    return HttpResponse.json({
      items: rows.slice(offset, offset + limit).map((data, i) => ({
        line: physicalLine(path, offset + i),
        id: typeof data["id"] === "string" ? data["id"] : null,
        data,
        error: bad.get(offset + i + 1) ?? null,
      })),
      total: rows.length,
      offset,
      limit,
      path,
      kind: s.kind,
      modified_at: s.modified_at,
      writable: s.writable,
      // Load-time errors name physical file lines; validation reports use row ordinals.
      errors: rep.errors.map((e) => ({ line: physicalLine(path, e.line - 1), message: e.message })),
    });
  }),
  http.put("/api/datasets/*", async ({ params, request }) => {
    const path = [params["0"]].flat().join("/");
    const s = summaryFor(path);
    const q = new URL(request.url).searchParams;
    const b = (await request.json()) as DatasetWrite;
    if (!path.endsWith(".jsonl")) return err(422, "invalid", "only .jsonl datasets can be written");
    if (!s && q.get("create") !== "true") return err(404, "not_found", "no such dataset");
    if (s && q.get("create") === "true") return err(409, "conflict", "dataset already exists");
    if (s && !s.writable) return err(403, "forbidden", "committed file; only *.sample.jsonl in examples/ or gitignored paths are editable");
    if (s && b.expected_modified_at && b.expected_modified_at !== s.modified_at) return err(409, "conflict", "the file changed on disk since it was loaded");
    const rep = validate(b.rows, b.kind ?? s?.kind ?? "golden");
    if (q.get("dry_run") === "true") return HttpResponse.json(rep);
    if (!rep.ok) return err(422, "case_format", `${path}: line ${rep.errors[0]?.line ?? 0}: ${rep.errors[0]?.message ?? "invalid"}`, true);
    datasetState.set(path, b.rows);
    if (s) {
      s.modified_at = new Date().toISOString();
      s.rows = b.rows.length;
      s.invalid_rows = 0;
      s.review_required = b.rows.filter((r) => r["review_required"]).length;
    } else {
      createdDatasets.add(path);
      DATASETS.push({ path, agent_id: null, kind: b.kind ?? "golden", rows: b.rows.length, invalid_rows: 0, review_required: 0, writable: true, modified_at: new Date().toISOString() });
    }
    return HttpResponse.json(summaryFor(path));
  }),

  http.post("/api/runs", async ({ request }) => {
    const body = (await request.json()) as RunRequest;
    const agent = AGENTS.find((a) => a.id === body.agent);
    if (!agent) return err(404, "not_found", `unknown agent '${body.agent}'`);
    const health = mockState.health;
    if (!health.demo && !health.langsmith_key && !("local" in body && body.local) && body.command !== "harvest" && body.command !== "score") {
      return err(422, "preflight", "LANGSMITH_API_KEY is not set; export it, or pass --local to keep results offline", true);
    }
    if (!agent.valid) return err(422, "preflight", `cannot start agent '${agent.name}': ${agent.error?.message ?? "invalid configuration"}`, true);
    if ("data" in body && !DATASETS.some((d) => d.path === body.data)) return err(422, "case_format", `no such data file: ${body.data}`, true);
    const m = startJob(body);
    return HttpResponse.json(m.job, { status: 202, headers: { Location: `/api/runs/${m.job.id}` } });
  }),
  http.get("/api/runs", () => {
    ensureFixtureJob();
    return HttpResponse.json([...jobs.values()].map((m) => m.job).reverse());
  }),
  http.get("/api/runs/:id", ({ params }) => {
    ensureFixtureJob();
    const m = jobs.get(String(params["id"]));
    return m ? HttpResponse.json(m.job) : err(404, "not_found", "unknown job");
  }),
  http.delete("/api/runs/:id", ({ params }) => {
    const m = jobs.get(String(params["id"]));
    if (!m) return err(404, "not_found", "unknown job");
    const changed = cancelJob(m);
    return HttpResponse.json(m.job, { status: changed ? 202 : 200 });
  }),
  http.get("/api/runs/:id/events.json", ({ params, request }) => {
    const m = jobs.get(String(params["id"]));
    if (!m) return err(404, "not_found", "unknown job");
    const after = num(new URL(request.url).searchParams.get("after"), 0);
    return HttpResponse.json(m.events.filter((e) => e.id > after));
  }),
  http.get("/api/runs/:id/events", ({ params, request }) => {
    ensureFixtureJob();
    const m = jobs.get(String(params["id"]));
    if (!m) return err(404, "not_found", "unknown job");
    const url = new URL(request.url);
    const after = num(request.headers.get("Last-Event-ID") ?? url.searchParams.get("after"), 0);
    if (isTerminalJob(m) && after >= m.job.last_event_id) return new HttpResponse(null, { status: 204 });
    const enc = new TextEncoder();
    let sent = after;
    let unsub = () => undefined as void;
    const stream = new ReadableStream<Uint8Array>({
      start(ctl) {
        ctl.enqueue(enc.encode("retry: 3000\n\n"));
        const flush = () => {
          for (const e of m.events) {
            if (e.id <= sent) continue;
            sent = e.id;
            ctl.enqueue(enc.encode(`id: ${e.id}\nevent: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`));
            if (e.event === "completed" || e.event === "failed" || e.event === "cancelled") {
              unsub();
              ctl.close();
              return;
            }
          }
        };
        m.listeners.add(flush);
        unsub = () => void m.listeners.delete(flush);
        request.signal.addEventListener("abort", () => unsub());
        flush();
      },
      cancel() {
        unsub();
      },
    });
    return new HttpResponse(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } });
  }),

  http.get("/api/results", ({ request }) => {
    const p = new URL(request.url).searchParams;
    const agent = p.get("agent");
    const kind = p.get("kind");
    const items = RESULTS.filter((r) => (!agent || r.agent === agent) && (!kind || r.kind === kind));
    const offset = num(p.get("offset"), 0);
    const limit = num(p.get("limit"), 50);
    return HttpResponse.json({ items: items.slice(offset, offset + limit), total: items.length, offset, limit });
  }),
  http.get("/api/results/:id", ({ params }) => {
    const r = RESULTS.find((x) => x.run_id === params["id"]);
    return r ? HttpResponse.json(r) : err(404, "not_found", `unknown run '${String(params["id"])}'`);
  }),
  http.get("/api/results/:id/rows", ({ params, request }) => rowsHandler(String(params["id"]), undefined, new URL(request.url))),
  http.get("/api/results/:id/rows/:index", ({ params }) => rowHandler(String(params["id"]), undefined, String(params["index"]))),
  http.get("/api/results/:id/evals/:evalId", ({ params }) => {
    const d = EVAL_DETAILS[String(params["evalId"])];
    return d ? HttpResponse.json(d) : err(404, "not_found", "unknown eval");
  }),
  http.get("/api/results/:id/evals/:evalId/rows", ({ params, request }) => rowsHandler(String(params["id"]), String(params["evalId"]), new URL(request.url))),
  http.get("/api/results/:id/evals/:evalId/rows/:index", ({ params }) => rowHandler(String(params["id"]), String(params["evalId"]), String(params["index"]))),
  http.get("/api/results/:id/prompts/:candidate", ({ params }) => {
    const p = PROMPTS[String(params["candidate"])];
    return p ? HttpResponse.json(p) : err(404, "not_found", "unknown candidate");
  }),
  http.get("/api/compare", ({ request }) => {
    const p = new URL(request.url).searchParams;
    const { ra, rb } = comparison(p.get("a") ?? "", p.get("b") ?? "");
    if (!ra || !rb) return err(404, "not_found", "unknown run");
    if (ra.detail !== "experiment" || rb.detail !== "experiment") return err(422, "invalid", "compare needs two experiment runs");
    const keys = [...new Set([...Object.keys(ra.summary), ...Object.keys(rb.summary)])].filter((k) => !k.includes("["));
    const ids = (runId: string) => (seedsFor(runId) ?? []).map((s) => s.case_id);
    const status = (runId: string, c: string) => seedsFor(runId)?.find((s) => s.case_id === c)?.status;
    const common = ids(ra.run_id).filter((c) => ids(rb.run_id).includes(c));
    const fixed = common.filter((c) => status(ra.run_id, c) !== "passed" && status(rb.run_id, c) === "passed");
    const regressed = common.filter((c) => status(ra.run_id, c) === "passed" && status(rb.run_id, c) !== "passed");
    return HttpResponse.json({
      a: ra,
      b: rb,
      metrics: keys.map((key) => {
        const a = ra.summary[key] ?? null;
        const b = rb.summary[key] ?? null;
        return { key, a, b, delta: a !== null && b !== null ? +(b - a).toFixed(4) : null };
      }),
      fixed_cases: ra.pass_keys && rb.pass_keys ? fixed : null,
      regressed_cases: ra.pass_keys && rb.pass_keys ? regressed : null,
      sign_test_p: ra.pass_keys && rb.pass_keys ? 0.5 : null,
      only_in_a: ids(ra.run_id).filter((c) => !ids(rb.run_id).includes(c)),
      only_in_b: ids(rb.run_id).filter((c) => !ids(ra.run_id).includes(c)),
    });
  }),

  http.post("/api/langsmith/pull", async ({ request }) => {
    const b = (await request.json()) as { agent: string; dataset: string; out: string };
    if (!b.out.startsWith("data/")) return err(422, "invalid", "out must be under data/");
    if (b.dataset === "missing") return err(404, "langsmith_not_found", "dataset 'missing' was not found in LangSmith");
    if (summaryFor(b.out) || createdDatasets.has(b.out)) return err(409, "exists", "the output file already exists");
    return HttpResponse.json({ path: b.out }, { status: 201 });
  }),
  http.post("/api/langsmith/push", async ({ request }) => {
    const b = (await request.json()) as { agent: string; data: string; dataset?: string | null };
    const s = summaryFor(b.data);
    if (!s) return err(404, "not_found", "no such dataset");
    if (s.review_required > 0) return err(422, "invalid", `${b.data}: ${s.review_required} rows still have review_required set; approve or remove them first`, true);
    return HttpResponse.json({ path: b.data, name: b.dataset ?? `${b.agent.split(".").pop() ?? b.agent}/${(b.data.split("/").pop() ?? "").replace(/\.jsonl$/, "")}` });
  }),
];
