export type TranscriptItem =
  | { kind: "message"; role: "user" | "assistant" | "system"; text: string }
  | { kind: "tool"; id: string | null; name: string; args: string; result: string | null }
  | { kind: "unknown"; text: string };

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** Message content as plain text. Anything that is not a string or text parts is shown as JSON text. */
export function contentText(content: unknown): string {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => (isRecord(p) && typeof p["text"] === "string" ? p["text"] : JSON.stringify(p)))
      .join("\n");
  }
  return JSON.stringify(content, null, 2);
}

function prettyArgs(raw: unknown): string {
  if (typeof raw !== "string") return raw === undefined ? "" : JSON.stringify(raw, null, 2);
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

/**
 * Turns an OpenAI-format transcript into render items. Tool results are attached to the call with the same
 * `tool_call_id`; a result with no matching call stays visible as its own tool item. Output is plain strings only.
 */
export function parseTranscript(messages: unknown): TranscriptItem[] {
  if (!Array.isArray(messages)) return [];
  const items: TranscriptItem[] = [];
  const byId = new Map<string, Extract<TranscriptItem, { kind: "tool" }>>();
  for (const m of messages) {
    if (!isRecord(m)) {
      items.push({ kind: "unknown", text: JSON.stringify(m) });
      continue;
    }
    const role = m["role"];
    if (role === "tool") {
      const id = typeof m["tool_call_id"] === "string" ? m["tool_call_id"] : null;
      const call = id ? byId.get(id) : undefined;
      const text = contentText(m["content"]);
      if (call) call.result = text;
      else items.push({ kind: "tool", id, name: typeof m["name"] === "string" ? m["name"] : "tool", args: "", result: text });
      continue;
    }
    if (role === "user" || role === "assistant" || role === "system") {
      const text = contentText(m["content"]);
      if (text) items.push({ kind: "message", role, text });
      const calls = m["tool_calls"];
      if (role === "assistant" && Array.isArray(calls)) {
        for (const c of calls) {
          if (!isRecord(c)) continue;
          const fn = isRecord(c["function"]) ? c["function"] : {};
          const item: Extract<TranscriptItem, { kind: "tool" }> = {
            kind: "tool",
            id: typeof c["id"] === "string" ? c["id"] : null,
            name: typeof fn["name"] === "string" ? fn["name"] : "unknown tool",
            args: prettyArgs(fn["arguments"]),
            result: null,
          };
          if (item.id) byId.set(item.id, item);
          items.push(item);
        }
      }
      continue;
    }
    items.push({ kind: "unknown", text: JSON.stringify(m) });
  }
  return items;
}
