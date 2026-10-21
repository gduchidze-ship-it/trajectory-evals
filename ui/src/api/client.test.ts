import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { server } from "../mocks/server";
import { ApiError, api, encodePath, errorInfoFrom, request, withQuery } from "./client";

describe("withQuery and encodePath", () => {
  it("drops empty values and encodes the rest", () => {
    expect(withQuery("/x", { a: "b c", n: 0, skip: null, none: undefined, empty: "" })).toBe("/x?a=b+c&n=0");
    expect(withQuery("/x", {})).toBe("/x");
  });
  it("encodes each path segment but keeps the slashes", () => {
    expect(encodePath("examples/my agent/a#b.md")).toBe("examples/my%20agent/a%23b.md");
  });
});

describe("request", () => {
  it("returns the parsed body", async () => {
    const health = await api.health();
    expect(health.status).toBe("ok");
  });

  it("sends JSON bodies with a JSON content type", async () => {
    let seen: { type: string | null; body: unknown } | null = null;
    server.use(
      http.post("/api/agents/:id/check", async ({ request: r }) => {
        seen = { type: r.headers.get("content-type"), body: await r.json() };
        return HttpResponse.json({ agent_id: "a", ok: true, steps: [] });
      }),
    );
    await api.check("examples.support_agent", true);
    expect(seen).toEqual({ type: "application/json", body: { ping: true } });
  });

  it("throws an ApiError carrying the ErrorBody message verbatim", async () => {
    server.use(
      http.get("/api/agents/:id", () =>
        HttpResponse.json({ error: { kind: "agent_config", message: "…/agent.toml: [judges.helpful] prompt may use only …", input_error: true } }, { status: 422 }),
      ),
    );
    const err = await api.agent("x").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const e = err as ApiError;
    expect(e.status).toBe(422);
    expect(e.message).toBe("…/agent.toml: [judges.helpful] prompt may use only …");
    expect(e.info).toEqual({ kind: "agent_config", message: e.message, input_error: true });
  });

  it("does not echo a non-ErrorBody response", async () => {
    server.use(http.get("/api/health", () => new HttpResponse("<html>boom customer text</html>", { status: 500 })));
    const e = (await api.health().catch((x: unknown) => x)) as ApiError;
    expect(e.info.kind).toBe("internal");
    expect(e.message).toBe("The server answered HTTP 500.");
  });

  it("treats 204 as no content", async () => {
    server.use(http.get("/api/thing", () => new HttpResponse(null, { status: 204 })));
    await expect(request<undefined>("GET", "/thing")).resolves.toBeUndefined();
  });

  it("puts dry_run on the query string for file validation", async () => {
    let url = "";
    server.use(
      http.put("/api/agents/:id/files/*", ({ request: r }) => {
        url = r.url;
        return HttpResponse.json({ path: "agent.toml", text: "" });
      }),
    );
    await api.writeFile("examples.support_agent", "prompts/system.md", "hi", true);
    expect(new URL(url).pathname).toBe("/api/agents/examples.support_agent/files/prompts/system.md");
    expect(new URL(url).searchParams.get("dry_run")).toBe("true");
  });
});

describe("errorInfoFrom", () => {
  it("accepts a bare ErrorInfo and fills input_error", () => {
    expect(errorInfoFrom(403, { kind: "forbidden", message: "host not allowed" })).toEqual({ kind: "forbidden", message: "host not allowed", input_error: false });
  });
});
