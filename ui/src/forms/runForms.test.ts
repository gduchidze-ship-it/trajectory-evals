import { describe, expect, it } from "vitest";
import { COMMANDS } from "../api/types";
import { buildRequest, defaultValues, validateForm } from "./runForms";

describe("defaults", () => {
  it("match the RunRequest model defaults", () => {
    expect(defaultValues("eval")).toMatchObject({ local: true, repetitions: "1", concurrency: "4", limit: "" });
    expect(defaultValues("harvest")["limit"]).toBe("100");
    expect(defaultValues("score")["dry_run"]).toBe(true);
    expect(defaultValues("redteam")["personas"]).toEqual([]);
  });
});

describe("validateForm", () => {
  it("requires agent and data for eval", () => {
    expect(validateForm("eval", defaultValues("eval"))).toEqual({ agent: "Agent is required.", data: "Golden dataset is required." });
  });
  it("does not require data for red-team", () => {
    expect(validateForm("redteam", { ...defaultValues("redteam"), agent: "examples.support_agent" })).toEqual({});
  });
  it("rejects non-positive and fractional integers", () => {
    const base = { ...defaultValues("eval"), agent: "a", data: "d" };
    expect(validateForm("eval", { ...base, limit: "0" })["limit"]).toMatch(/whole number/);
    expect(validateForm("eval", { ...base, concurrency: "1.5" })["concurrency"]).toMatch(/whole number/);
    expect(validateForm("eval", { ...base, limit: "3" })).toEqual({});
  });
  it("rejects local plus upload for optimize", () => {
    const v = { ...defaultValues("optimize"), agent: "a", data: "d", local: true, upload_as_experiments: true };
    expect(validateForm("optimize", v)["upload_as_experiments"]).toMatch(/not both/);
  });
  it("needs a project when the agent has none", () => {
    const v = { ...defaultValues("harvest"), agent: "a" };
    expect(validateForm("harvest", v, { projectConfigured: false })["project"]).toMatch(/no \[langsmith\]\.project/);
    expect(validateForm("harvest", v, { projectConfigured: true })).toEqual({});
    expect(validateForm("harvest", { ...v, project: "dev" }, { projectConfigured: false })).toEqual({});
  });
  it("checks feedback_below is a number", () => {
    const v = { ...defaultValues("harvest"), agent: "a", feedback_below: "abc" };
    expect(validateForm("harvest", v)["feedback_below"]).toMatch(/number/);
  });
});

describe("buildRequest", () => {
  it("produces the contract's eval body", () => {
    expect(buildRequest("eval", { ...defaultValues("eval"), agent: "examples.support_agent", data: "examples/support_agent/datasets/golden.sample.jsonl", limit: "3" })).toEqual({
      command: "eval",
      agent: "examples.support_agent",
      data: "examples/support_agent/datasets/golden.sample.jsonl",
      local: true,
      limit: 3,
      repetitions: 1,
      concurrency: 4,
      prefix: null,
      criteria_model: null,
    });
  });
  it("maps an empty persona selection to null and keeps a chosen one", () => {
    expect(buildRequest("redteam", { ...defaultValues("redteam"), agent: "a" })).toMatchObject({ personas: null, attempts: null });
    expect(buildRequest("redteam", { ...defaultValues("redteam"), agent: "a", personas: ["prompt_injector"] })).toMatchObject({ personas: ["prompt_injector"] });
  });
  it("builds the harvest body from the contract's example", () => {
    expect(buildRequest("harvest", { ...defaultValues("harvest"), agent: "examples.sequence_runner_sales", project: "dev-sales", since_days: "3", limit: "50", only_errors: true })).toEqual({
      command: "harvest",
      agent: "examples.sequence_runner_sales",
      project: "dev-sales",
      since_days: 3,
      limit: 50,
      filter: null,
      only_errors: true,
      feedback_key: null,
      feedback_below: null,
      out: null,
    });
  });
  it("builds every command without throwing from defaults", () => {
    for (const c of COMMANDS) expect(buildRequest(c, { ...defaultValues(c), agent: "a", data: "d" }).command).toBe(c);
  });
});
