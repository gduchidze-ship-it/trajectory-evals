import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { parseTranscript } from "../lib/transcript";
import { Transcript } from "./Transcript";

const EVIL = `<script>window.__pwned = 1</script><img src=x onerror="window.__pwned = 2"> **bold** [x](javascript:alert(1))`;

const messages = [
  { role: "user", content: EVIL },
  { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "lookup_order", arguments: '{"order_id":"A1001"}' } }] },
  { role: "tool", tool_call_id: "c1", content: "<b>delivered</b>" },
  { role: "assistant", content: "Done." },
  { role: "tool", tool_call_id: "ghost", name: "stray", content: "orphan result" },
];

describe("parseTranscript", () => {
  it("groups a tool result under its call and keeps orphan results visible", () => {
    const items = parseTranscript(messages);
    expect(items.map((i) => i.kind)).toEqual(["message", "tool", "message", "tool"]);
    expect(items[1]).toMatchObject({ kind: "tool", name: "lookup_order", result: "<b>delivered</b>" });
    expect((items[1] as { args: string }).args).toContain('"order_id": "A1001"');
    expect(items[3]).toMatchObject({ kind: "tool", name: "stray", result: "orphan result" });
  });
  it("copes with junk", () => {
    expect(parseTranscript(null)).toEqual([]);
    expect(parseTranscript([42, { role: "wat" }]).map((i) => i.kind)).toEqual(["unknown", "unknown"]);
  });
});

describe("Transcript", () => {
  it("renders markup in messages as inert text", () => {
    const { container } = render(<Transcript messages={messages} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("a")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
    expect(container.textContent).toContain("<script>window.__pwned = 1</script>");
    expect(container.textContent).toContain("**bold**");
    expect((window as unknown as { __pwned?: number }).__pwned).toBeUndefined();
  });

  it("shows tool calls and tool results distinctly from chat messages", () => {
    const { container } = render(<Transcript messages={messages} />);
    expect(container.querySelectorAll('[data-kind="message"]')).toHaveLength(2);
    const tool = container.querySelector('[data-kind="tool"]') as HTMLElement;
    expect(within(tool).getByText("lookup_order")).toBeInTheDocument();
    expect(tool.querySelector('[data-kind="tool-args"]')).toHaveTextContent('"A1001"');
    expect(tool.querySelector('[data-kind="tool-result"]')).toHaveTextContent("<b>delivered</b>");
    expect(container.querySelectorAll('[data-kind="message"] [data-kind="tool"]')).toHaveLength(0);
  });

  it("says so when there is nothing to show", () => {
    render(<Transcript messages={[]} />);
    expect(screen.getByText("This row has no transcript.")).toBeInTheDocument();
  });
});
