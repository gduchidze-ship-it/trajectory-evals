import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { makeQueryClient, routes } from "../App";
import { api } from "../api/client";
import { mockState } from "../mocks/fixtures";

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

describe("shell", () => {
  it("shows the demo banner when Health.demo is set, and not otherwise", async () => {
    const { unmount } = renderAt("/agents");
    expect(await screen.findByText(/scripted models, nothing leaves this machine/)).toBeInTheDocument();
    unmount();
    mockState.health.demo = false;
    renderAt("/agents");
    await screen.findByText("support_agent");
    expect(screen.queryByText(/scripted models/)).toBeNull();
  });
});

describe("agents", () => {
  it("lists health verbatim, with the error kind", async () => {
    renderAt("/agents");
    expect(await screen.findByText("…/agent.toml: [graders] No module named 'src'")).toBeInTheDocument();
    expect(screen.getAllByText("agent_config").length).toBeGreaterThan(0);
  });
});

describe("launcher", () => {
  it("validates before posting and highlights the missing fields", async () => {
    const user = userEvent.setup();
    renderAt("/launch");
    await user.click(await screen.findByRole("button", { name: /launch eval/i }));
    expect(await screen.findByText("Agent is required.", { selector: "p" })).toBeInTheDocument();
    expect(screen.getByLabelText("Agent")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("alert")).toHaveTextContent("Fix these 2 fields");
  });

  it("forces local on and disables upload when there is no LangSmith key", async () => {
    mockState.health.demo = false;
    mockState.health.langsmith_key = false;
    const user = userEvent.setup();
    renderAt("/launch?command=optimize");
    const box = await screen.findByLabelText("Upload candidates as LangSmith experiments");
    expect(box).toBeDisabled();
    await user.selectOptions(await screen.findByLabelText("Agent"), "examples.support_agent");
    expect(screen.getByLabelText("Keep results local")).toBeChecked();
    expect(screen.getAllByText(/LANGSMITH_API_KEY is not set/).length).toBeGreaterThan(0);
  });

  it("starts a run and follows it live to completion", async () => {
    const user = userEvent.setup();
    renderAt("/launch");
    await user.selectOptions(await screen.findByLabelText("Agent"), "examples.support_agent");
    await user.selectOptions(await screen.findByLabelText("Golden dataset"), "examples/support_agent/datasets/golden.sample.jsonl");
    await user.click(screen.getByRole("button", { name: /launch eval/i }));
    expect(await screen.findByRole("link", { name: "Open results" }, { timeout: 5000 })).toHaveAttribute("href", expect.stringContaining("/results/"));
    expect(screen.getAllByText("refund-eligible").length).toBeGreaterThan(0);
  });
});

describe("langsmith", () => {
  it("disables actions with an explanation when there is no key and no demo", async () => {
    mockState.health.demo = false;
    renderAt("/langsmith");
    expect(await screen.findByText(/LangSmith actions are disabled/)).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Start harvest" })).toBeDisabled();
  });

  it("is enabled in demo mode", async () => {
    renderAt("/langsmith");
    await screen.findByText(/in-memory LangSmith stand-in/);
    await waitFor(() => expect(screen.getByRole("button", { name: "Start harvest" })).toBeEnabled());
  });

  it("push refuses rows under review and says where to go", async () => {
    const user = userEvent.setup();
    renderAt("/langsmith?tab=push");
    await user.selectOptions(await screen.findByLabelText("Agent"), "examples.sales_assistant");
    await user.selectOptions(screen.getByLabelText("Golden dataset"), "data/langsmith/sales_assistant-20261021_120000.jsonl");
    await user.click(screen.getByRole("button", { name: "Push dataset" }));
    expect(await screen.findByText(/still have review_required set/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open the review queue" })).toBeInTheDocument();
  });
});

describe("results", () => {
  it("shows the pass rate with its Wilson interval and filters rows by status", async () => {
    const user = userEvent.setup();
    renderAt("/results/20261021_114757_support-agent-eval-local_f0736f");
    expect(await screen.findByText(/95% CI 20\.8% to 93\.8%/)).toBeInTheDocument();
    expect(screen.getAllByText("66.7%").length).toBeGreaterThan(0);
    expect(await screen.findByText("lie-about-refund")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^Failed/ }));
    await waitFor(() => expect(screen.queryByText("refund-eligible")).toBeNull());
    expect(screen.getByText("lie-about-refund")).toBeInTheDocument();
  });

  it("opens a transcript drawer that renders hostile text inertly", async () => {
    const user = userEvent.setup();
    renderAt("/results/20261021_113000_support-agent-simulate-local_c4d2e8?segment=persona%3Dprompt_injector");
    await user.click(await screen.findByText("prompt_injector-1"));
    const dlg = await screen.findByRole("dialog");
    await waitFor(() => expect(dlg.textContent).toContain("<script>"));
    expect(dlg.querySelector("script")).toBeNull();
    expect(dlg.querySelector("img")).toBeNull();
    expect(dlg.querySelectorAll('[data-kind="tool"]').length).toBeGreaterThan(0);
  });

  it("surfaces the ASR coverage warning for a red-team run", async () => {
    renderAt("/results/20261021_112200_support-agent-redteam-local_5e6f70");
    expect(await screen.findByText(/The attack success rate is uncertain/)).toBeInTheDocument();
  });

  it("shows an unknown run as a not-found state", async () => {
    renderAt("/results/nope");
    expect(await screen.findByText("No such run")).toBeInTheDocument();
  });
});

describe("compare", () => {
  it("lists fixed cases with drill-down links", async () => {
    renderAt("/compare?a=20261021_101200_support-agent-eval-local_3b7e10&b=20261021_121530_support-agent-eval-local_a91c2d");
    const fixed = await screen.findByRole("region", { name: "Fixed" });
    expect(fixed).toHaveTextContent("late-delivery");
    expect(fixed.querySelectorAll("a")).toHaveLength(2);
  });
});

describe("optimizer report", () => {
  it("shows lineage, floors and the holdout verdict", async () => {
    renderAt("/results/20261021_105500_support-agent-optimize-local_9a8b7c");
    const c2 = await screen.findByTestId("candidate-candidate-2");
    expect(c2).toHaveTextContent("rejected");
    expect(c2).toHaveTextContent("regressed high-risk cases");
    expect(c2.querySelector('[aria-label="floor failed"]')).not.toBeNull();
    expect(await screen.findByText(/sign test p/i, { selector: "div" })).toBeInTheDocument();
    expect(screen.getAllByText("adopt").length).toBeGreaterThan(0);
  });

  it("marks an unfinished run", async () => {
    renderAt("/results/20261021_103000_support-agent-optimize-local_77aa00");
    expect(await screen.findByText(/In progress or interrupted/)).toBeInTheDocument();
  });
});

describe("api smoke", () => {
  it("cancel is idempotent against the mock", async () => {
    const job = await api.startRun({ command: "eval", agent: "examples.support_agent", data: "examples/support_agent/datasets/golden.sample.jsonl", local: true, limit: 3, repetitions: 1, concurrency: 4, prefix: null, criteria_model: null });
    const first = await api.cancelRun(job.id);
    const second = await api.cancelRun(job.id);
    expect(first.state).toBe("cancelled");
    expect(second.state).toBe("cancelled");
  });
});
