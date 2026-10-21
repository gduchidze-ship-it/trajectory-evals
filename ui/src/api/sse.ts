import { ApiError, errorInfoFrom, API_BASE } from "./client";
import type { Cancelled, Completed, EventName, ExperimentStarted, Failed, LogLine, RowDone, StateChanged } from "./types";

/** One SSE frame; `data` is the payload model, not the RunEvent envelope (contract B.3). */
export type StreamEvent =
  | { id: number; event: "state"; data: StateChanged }
  | { id: number; event: "experiment"; data: ExperimentStarted }
  | { id: number; event: "row"; data: RowDone }
  | { id: number; event: "log"; data: LogLine }
  | { id: number; event: "completed"; data: Completed }
  | { id: number; event: "failed"; data: Failed }
  | { id: number; event: "cancelled"; data: Cancelled };

const EVENT_NAMES: readonly EventName[] = ["state", "experiment", "row", "log", "completed", "failed", "cancelled"];
const isEventName = (s: string): s is EventName => (EVENT_NAMES as readonly string[]).includes(s);

export const isTerminal = (e: StreamEvent): boolean =>
  e.event === "completed" || e.event === "failed" || e.event === "cancelled";

interface RawFrame {
  id: string | null;
  event: string;
  data: string;
  retry: number | null;
}

/** Incremental text/event-stream parser. Feed it chunks; it returns the frames completed by each chunk. */
export class SseParser {
  private buffer = "";
  push(chunk: string): RawFrame[] {
    this.buffer += chunk;
    const frames: RawFrame[] = [];
    const parts = this.buffer.split(/\r\n\r\n|\n\n|\r\r/);
    this.buffer = parts.pop() ?? "";
    for (const part of parts) {
      const frame: RawFrame = { id: null, event: "message", data: "", retry: null };
      const data: string[] = [];
      let any = false;
      for (const line of part.split(/\r\n|\n|\r/)) {
        if (line === "" || line.startsWith(":")) continue;
        const i = line.indexOf(":");
        const field = i === -1 ? line : line.slice(0, i);
        const value = i === -1 ? "" : line.slice(i + 1).replace(/^ /, "");
        if (field === "id") frame.id = value;
        else if (field === "event") frame.event = value;
        else if (field === "data") data.push(value);
        else if (field === "retry" && /^\d+$/.test(value)) frame.retry = Number(value);
        else continue;
        any = true;
      }
      frame.data = data.join("\n");
      if (any) frames.push(frame);
    }
    return frames;
  }
}

function toStreamEvent(f: RawFrame): StreamEvent | null {
  if (f.id === null || !/^\d+$/.test(f.id) || !isEventName(f.event)) return null;
  let data: unknown;
  try {
    data = f.data ? JSON.parse(f.data) : {};
  } catch {
    return null;
  }
  if (typeof data !== "object" || data === null) return null;
  // The server serialises the payload model for exactly this event name; the discriminant is the SSE `event:` field.
  return { id: Number(f.id), event: f.event, data } as StreamEvent;
}

export interface FollowOptions {
  jobId: string;
  onEvent: (e: StreamEvent) => void;
  onConnection?: (s: "live" | "reconnecting") => void;
  signal: AbortSignal;
  fetchImpl?: typeof fetch;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  startAfter?: number;
}

const DEFAULT_RETRY_MS = 3000;

const abortableSleep = (ms: number, signal: AbortSignal): Promise<void> =>
  new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(t), resolve()), { once: true });
  });

/**
 * Follows a job's event stream until a terminal event, a 204, or abort. Reconnects after a dropped stream with
 * both `Last-Event-ID` and `?after=` set to the last id seen, and drops replayed ids it already delivered.
 */
export async function followRun(o: FollowOptions): Promise<"terminal" | "no-content" | "aborted"> {
  const doFetch = o.fetchImpl ?? fetch;
  const sleep = o.sleep ?? abortableSleep;
  let lastId = o.startAfter ?? 0;
  let retryMs = DEFAULT_RETRY_MS;
  for (;;) {
    if (o.signal.aborted) return "aborted";
    try {
      const res = await doFetch(`${API_BASE}/runs/${o.jobId}/events?after=${lastId}`, {
        headers: { Accept: "text/event-stream", "Last-Event-ID": String(lastId) },
        signal: o.signal,
      });
      if (res.status === 204) return "no-content";
      if (!res.ok || !res.body) {
        const body: unknown = await res.json().catch(() => null);
        throw new ApiError(res.status, errorInfoFrom(res.status, body));
      }
      o.onConnection?.("live");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const parser = new SseParser();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const raw of parser.push(decoder.decode(value, { stream: true }))) {
          if (raw.retry !== null) retryMs = raw.retry;
          const ev = toStreamEvent(raw);
          if (!ev || ev.id <= lastId) continue;
          lastId = ev.id;
          o.onEvent(ev);
          if (isTerminal(ev)) {
            await reader.cancel().catch(() => undefined);
            return "terminal";
          }
        }
      }
    } catch (e) {
      if (o.signal.aborted) return "aborted";
      if (e instanceof ApiError && e.status >= 400 && e.status < 500) throw e;
    }
    o.onConnection?.("reconnecting");
    await sleep(retryMs, o.signal);
  }
}
