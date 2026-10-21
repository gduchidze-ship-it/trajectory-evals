import { describe, expect, it, vi } from "vitest";
import { ApiError } from "./client";
import { followRun, SseParser, type StreamEvent } from "./sse";

const enc = new TextEncoder();
const frame = (id: number, event: string, data: unknown) => `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

function streamOf(chunks: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      for (const ch of chunks) c.enqueue(enc.encode(ch));
      c.close();
    },
  });
  return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
}

const row = (n: number) => ({ experiment: "e", done: n, total: 9, case_id: `c${n}`, status: "passed", latency_ms: 1, feedback: {} });

describe("SseParser", () => {
  it("assembles frames split across chunks, ignores comments, reads retry", () => {
    const p = new SseParser();
    expect(p.push("retry: 3000\n\n: ping\n\nid: 1\nev")).toHaveLength(1);
    const out = p.push('ent: row\ndata: {"a":1}\n\nid: 2\nevent: log\ndata: {"b"');
    expect(out).toEqual([{ id: "1", event: "row", data: '{"a":1}', retry: null }]);
    expect(p.push(":2}\n\n")).toEqual([{ id: "2", event: "log", data: '{"b":2}', retry: null }]);
  });
  it("handles CRLF line endings and multi-line data", () => {
    const frames = new SseParser().push("id: 5\r\nevent: log\r\ndata: a\r\ndata: b\r\n\r\n");
    expect(frames[0]?.data).toBe("a\nb");
  });
});

describe("followRun", () => {
  it("reconnects with Last-Event-ID and ?after=n, drops replays, and stops at the terminal event", async () => {
    const calls: { url: string; lastId: string | null }[] = [];
    const responses = [
      () => streamOf(["retry: 10\n\n", frame(1, "state", { state: "running" }), frame(2, "row", row(1))]),
      () => streamOf([frame(2, "row", row(1)), frame(3, "row", row(2)), frame(4, "completed", { result: { kind: "harvest", path: "p", rows: 1 } })]),
    ];
    const fetchImpl = vi.fn((url: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(url), lastId: new Headers(init?.headers).get("Last-Event-ID") });
      const next = responses.shift();
      if (!next) throw new Error("fetched after the terminal event");
      return Promise.resolve(next());
    });
    const got: StreamEvent[] = [];
    const sleeps: number[] = [];
    const outcome = await followRun({
      jobId: "abc123abc123",
      signal: new AbortController().signal,
      fetchImpl: fetchImpl as typeof fetch,
      sleep: (ms) => (sleeps.push(ms), Promise.resolve()),
      onEvent: (e) => got.push(e),
    });
    expect(outcome).toBe("terminal");
    expect(got.map((e) => e.id)).toEqual([1, 2, 3, 4]);
    expect(calls).toEqual([
      { url: "/api/runs/abc123abc123/events?after=0", lastId: "0" },
      { url: "/api/runs/abc123abc123/events?after=2", lastId: "2" },
    ]);
    expect(sleeps).toEqual([10]);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("reconnects after a network error", async () => {
    let n = 0;
    const fetchImpl = vi.fn(() => {
      n += 1;
      if (n === 1) return Promise.reject(new TypeError("network down"));
      return Promise.resolve(streamOf([frame(1, "cancelled", {})]));
    });
    const got: StreamEvent[] = [];
    const out = await followRun({ jobId: "j", signal: new AbortController().signal, fetchImpl: fetchImpl as typeof fetch, sleep: () => Promise.resolve(), onEvent: (e) => got.push(e) });
    expect(out).toBe("terminal");
    expect(got[0]?.event).toBe("cancelled");
  });

  it("stops on 204 without reconnecting", async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(new Response(null, { status: 204 })));
    const out = await followRun({ jobId: "j", signal: new AbortController().signal, fetchImpl: fetchImpl as typeof fetch, sleep: () => Promise.resolve(), onEvent: () => undefined, startAfter: 9 });
    expect(out).toBe("no-content");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("throws on 404 instead of retrying forever", async () => {
    const body = JSON.stringify({ error: { kind: "not_found", message: "unknown job", input_error: false } });
    const fetchImpl = vi.fn(() => Promise.resolve(new Response(body, { status: 404 })));
    await expect(followRun({ jobId: "j", signal: new AbortController().signal, fetchImpl: fetchImpl as typeof fetch, sleep: () => Promise.resolve(), onEvent: () => undefined })).rejects.toBeInstanceOf(ApiError);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("returns aborted when the signal fires", async () => {
    const ctl = new AbortController();
    ctl.abort();
    const out = await followRun({ jobId: "j", signal: ctl.signal, fetchImpl: vi.fn() as unknown as typeof fetch, onEvent: () => undefined });
    expect(out).toBe("aborted");
  });
});
