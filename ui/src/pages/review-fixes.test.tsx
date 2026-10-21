import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { makeQueryClient, routes } from "../App";
import { emptyCounts, initialRun, MAX_ROWS, reduceRun, runningPassRate } from "../api/runState";
import { server } from "../mocks/server";
import { errorsFromLines, errorsFromOrdinals } from "./DatasetEditor";
import { FeedbackChips } from "./ResultDetail";

vi.mock("../components/CodeEditor", () => ({
  CodeEditor: ({ value, label }: { value: string; label: string }) => <textarea aria-label={label} defaultValue={value} />,
  CodeDiff: () => <div />,
}));

function renderAt(path: string) {
  return render(
    <QueryClientProvider client={makeQueryClient()}>
      <RouterProvider router={createMemoryRouter(routes, { initialEntries: [path] })} />
    </QueryClientProvider>,
  );
}

describe("FeedbackChips", () => {
  it("shows a successful attack as a failure, not a green check", () => {
    const { container } = render(<FeedbackChips feedback={{ attack_succeeded: true, defended: false }} />);
    const [attack, defended] = Array.from(container.querySelectorAll(".chip"));
    expect(attack).toHaveClass("chip-fail");
    expect(defended).toHaveClass("chip-fail");
  });
  it("shows a defended attack as a pass, and other booleans normally", () => {
    const { container } = render(<FeedbackChips feedback={{ attack_succeeded: false, helpful: true, tone: null }} />);
    const chips = Array.from(container.querySelectorAll(".chip"));
    expect(chips[0]).toHaveClass("chip-pass");
    expect(chips[1]).toHaveClass("chip-pass");
    expect(chips[2]).not.toHaveClass("chip-pass");
    expect(chips[2]).not.toHaveClass("chip-fail");
  });
});

describe("dataset line numbers", () => {
  const entries = [1, 2, 3, 5, 6].map((line) => ({ line, data: {} }));
  it("maps load-time errors, which are physical lines, through each row's line, and keeps the ones with no row", () => {
    expect(errorsFromLines(entries, [{ line: 5, message: "x" }, { line: 4, message: "line 4 is not valid JSON" }])).toEqual({
      indexed: [{ index: 3, message: "x" }],
      unmapped: [{ line: 4, message: "line 4 is not valid JSON" }],
    });
  });
  it("maps validation errors, which are ordinals, to row indexes", () => {
    expect(errorsFromOrdinals([{ line: 4, message: "x" }])).toEqual([{ index: 3, message: "x" }]);
  });
  it("puts the badge on the right row when the file has a blank line", async () => {
    const user = userEvent.setup();
    renderAt("/datasets/examples/support_agent/datasets/golden.sample.jsonl");
    // The mock file has a blank physical line 4, so the bad seventh row is on physical line 8.
    const jump = await screen.findByRole("button", { name: "Jump to line 8" });
    await user.click(jump);
    const side = screen.getByRole("complementary", { name: "Row editor" });
    expect(side).toHaveTextContent("Line 8");
    const rows = screen.getAllByRole("row");
    const bad = rows.filter((r) => within(r).queryByText("error"));
    expect(bad).toHaveLength(1);
    expect(within(bad[0]!).getAllByRole("cell")[0]).toHaveTextContent("8");
  });
});

describe("dataset loading and saving", () => {
  it("refuses a file whose total changes between pages", async () => {
    server.use(
      http.get("/api/datasets/*", ({ request }) => {
        const offset = Number(new URL(request.url).searchParams.get("offset") ?? 0);
        return HttpResponse.json({
          items: [{ line: 1, id: "a", data: { id: "a", input: "x" }, error: null }],
          total: offset === 0 ? 2 : 3,
          offset: 0,
          limit: 500,
          path: "p.jsonl",
          kind: "golden",
          errors: [],
        });
      }),
    );
    renderAt("/datasets/examples/support_agent/datasets/golden.sample.jsonl");
    expect(await screen.findByText(/The file changed while it was loading/, undefined, { timeout: 5000 })).toBeInTheDocument();
  });

  it("on a 409 says the file changed and never claims it saved", async () => {
    const user = userEvent.setup();
    let sent: { expected_modified_at?: string } | null = null;
    server.use(
      http.put("/api/datasets/*", async ({ request }) => {
        sent = (await request.json()) as { expected_modified_at?: string };
        return HttpResponse.json({ error: { kind: "conflict", message: "the file changed on disk", input_error: false } }, { status: 409 });
      }),
    );
    renderAt("/datasets/examples/support_agent/datasets/scenarios.sample.jsonl");
    await user.click(await screen.findByRole("button", { name: /Add row/ }));
    const save = screen.getByRole("button", { name: /Save file/ });
    await waitFor(() => expect(save).toBeEnabled());
    await user.click(save);
    expect(await screen.findByText(/File changed on disk, reload/)).toBeInTheDocument();
    expect(sent).toMatchObject({ expected_modified_at: "2026-10-20T09:12:00Z" });
    expect(screen.queryByText("Saved.")).toBeNull();
  });
});

describe("after a save", () => {
  it("keeps the editor and the saved baseline when re-reading the file fails", async () => {
    const user = userEvent.setup();
    renderAt("/datasets/examples/support_agent/datasets/scenarios.sample.jsonl");
    await user.click(await screen.findByRole("button", { name: /Add row/ }));
    const save = screen.getByRole("button", { name: /Save file/ });
    await waitFor(() => expect(save).toBeEnabled());
    server.use(http.get("/api/datasets/*", () => HttpResponse.json({ error: { kind: "internal", message: "boom", input_error: false } }, { status: 500 })));
    await user.click(save);
    expect(await screen.findByText(/Saved\. Reloading the file from disk failed/, undefined, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Add row/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Save file/ })).toBeDisabled();
    expect(screen.queryByText("Could not load this dataset")).toBeNull();
  });
});

describe("screen errors", () => {
  it("keep the sidebar when a screen throws", async () => {
    server.use(http.get("/api/results/:id", () => HttpResponse.json({ detail: "experiment", run_id: "x" })));
    renderAt("/results/x");
    expect(await screen.findByText("This screen failed to render")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Main" })).toBeInTheDocument();
  });
});

describe("run state", () => {
  const row = (i: number, status: "passed" | "failed") => ({ id: i, event: "row" as const, data: { experiment: "e", done: i, total: 9999, case_id: `c${i}`, status, latency_ms: 1, feedback: {} } });
  it("caps the rows it keeps while counting every one", () => {
    let s = initialRun();
    for (let i = 1; i <= MAX_ROWS + 50; i++) s = reduceRun(s, row(i, i % 4 === 0 ? "failed" : "passed"));
    expect(s.rows).toHaveLength(MAX_ROWS);
    expect(s.rows[0]?.case_id).toBe("c51");
    expect(s.counts.passed + s.counts.failed).toBe(MAX_ROWS + 50);
    expect(runningPassRate(s.counts)).toBeCloseTo(0.75, 2);
  });
  it("has no pass rate before a scored row", () => {
    expect(runningPassRate(emptyCounts())).toBeNull();
  });
});
