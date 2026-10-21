import type {
  AgentDetail,
  AgentSummary,
  CheckReport,
  Comparison,
  DatasetPage,
  DatasetSummary,
  DatasetWrite,
  ErrorInfo,
  FileContent,
  Health,
  Page,
  PullRequest,
  PullResult,
  PushRequest,
  PushResult,
  ResultDetail,
  ResultKind,
  ResultSummary,
  RowDetail,
  RowStatus,
  RowSummary,
  RunEvent,
  RunJob,
  RunRequest,
  ValidationReport,
  ExperimentDetail,
  DatasetKind,
} from "./types";

export const API_BASE = "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly info: ErrorInfo;
  constructor(status: number, info: ErrorInfo) {
    super(info.message);
    this.name = "ApiError";
    this.status = status;
    this.info = info;
  }
}

function isErrorInfo(v: unknown): v is ErrorInfo {
  if (typeof v !== "object" || v === null) return false;
  const o = v as Record<string, unknown>;
  return typeof o["kind"] === "string" && typeof o["message"] === "string";
}

/** Body of an error response as ErrorInfo; anything that is not an ErrorBody gets a generic message. */
export function errorInfoFrom(status: number, body: unknown): ErrorInfo {
  if (typeof body === "object" && body !== null) {
    const inner = (body as Record<string, unknown>)["error"] ?? body;
    if (isErrorInfo(inner)) {
      return { kind: inner.kind, message: inner.message, input_error: inner.input_error === true };
    }
  }
  return { kind: "internal", message: `The server answered HTTP ${status}.`, input_error: false };
}

export function describeError(e: unknown): string {
  if (e instanceof ApiError) return e.info.message;
  if (e instanceof TypeError) return "Cannot reach the server. Is `python -m trajectory_evals ui` running?";
  return e instanceof Error ? e.message : "Unknown error";
}

export const encodePath = (p: string): string => p.split("/").map(encodeURIComponent).join("/");

type Query = Record<string, string | number | boolean | null | undefined>;

export function withQuery(path: string, query?: Query): string {
  if (!query) return path;
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  const s = sp.toString();
  return s ? `${path}?${s}` : path;
}

interface Options {
  query?: Query;
  body?: unknown;
  signal?: AbortSignal;
}

export async function request<T>(method: string, path: string, opts: Options = {}): Promise<T> {
  const init: RequestInit = { method, signal: opts.signal ?? null, headers: { Accept: "application/json" } };
  if (opts.body !== undefined) {
    init.body = JSON.stringify(opts.body);
    init.headers = { ...init.headers, "Content-Type": "application/json" };
  }
  const res = await fetch(withQuery(`${API_BASE}${path}`, opts.query), init);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }
  }
  if (!res.ok) throw new ApiError(res.status, errorInfoFrom(res.status, parsed));
  return parsed as T;
}

export interface RowQuery {
  offset?: number;
  limit?: number;
  status?: RowStatus | "";
  q?: string;
  segment?: string;
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>("GET", "/health", { signal }),
  agents: () => request<AgentSummary[]>("GET", "/agents"),
  agent: (id: string) => request<AgentDetail>("GET", `/agents/${encodeURIComponent(id)}`),
  check: (id: string, ping: boolean) =>
    request<CheckReport>("POST", `/agents/${encodeURIComponent(id)}/check`, { body: { ping } }),
  readFile: (id: string, path: string) =>
    request<FileContent>("GET", `/agents/${encodeURIComponent(id)}/files/${encodePath(path)}`),
  writeFile: (id: string, path: string, text: string, dryRun: boolean, signal?: AbortSignal) =>
    request<FileContent>("PUT", `/agents/${encodeURIComponent(id)}/files/${encodePath(path)}`, {
      query: { dry_run: dryRun },
      body: { text },
      signal,
    }),
  datasets: (agent?: string) => request<DatasetSummary[]>("GET", "/datasets", { query: { agent } }),
  dataset: (path: string, q: { offset?: number; limit?: number; kind?: DatasetKind }) =>
    request<DatasetPage>("GET", `/datasets/${encodePath(path)}`, { query: q }),
  validateRows: (body: DatasetWrite, signal?: AbortSignal) =>
    request<ValidationReport>("POST", "/datasets/validate", { body, signal }),
  writeDataset: (path: string, body: DatasetWrite, o: { create?: boolean; dryRun?: boolean } = {}) =>
    request<DatasetSummary | ValidationReport>("PUT", `/datasets/${encodePath(path)}`, {
      query: { create: o.create ?? false, dry_run: o.dryRun ?? false },
      body,
    }),
  startRun: (body: RunRequest) => request<RunJob>("POST", "/runs", { body }),
  runs: () => request<RunJob[]>("GET", "/runs"),
  run: (id: string, signal?: AbortSignal) => request<RunJob>("GET", `/runs/${id}`, { signal }),
  cancelRun: (id: string) => request<RunJob>("DELETE", `/runs/${id}`),
  runEvents: (id: string, after: number) => request<RunEvent[]>("GET", `/runs/${id}/events.json`, { query: { after } }),
  results: (q: { agent?: string; kind?: ResultKind | ""; offset?: number; limit?: number }) =>
    request<Page<ResultSummary>>("GET", "/results", { query: q }),
  result: (id: string) => request<ResultDetail>("GET", `/results/${encodeURIComponent(id)}`),
  evalDetail: (id: string, evalId: string) =>
    request<ExperimentDetail>("GET", `/results/${encodeURIComponent(id)}/evals/${encodeURIComponent(evalId)}`),
  rows: (id: string, q: RowQuery, evalId?: string) =>
    request<Page<RowSummary>>("GET", `${resultBase(id, evalId)}/rows`, { query: { ...q } }),
  row: (id: string, index: number, evalId?: string) =>
    request<RowDetail>("GET", `${resultBase(id, evalId)}/rows/${index}`),
  prompts: (id: string, candidate: string) =>
    request<Record<string, string>>("GET", `/results/${encodeURIComponent(id)}/prompts/${encodeURIComponent(candidate)}`),
  compare: (a: string, b: string) => request<Comparison>("GET", "/compare", { query: { a, b } }),
  pull: (body: PullRequest) => request<PullResult>("POST", "/langsmith/pull", { body }),
  push: (body: PushRequest) => request<PushResult>("POST", "/langsmith/push", { body }),
};

function resultBase(id: string, evalId?: string): string {
  const base = `/results/${encodeURIComponent(id)}`;
  return evalId ? `${base}/evals/${encodeURIComponent(evalId)}` : base;
}
