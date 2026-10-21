import clsx from "clsx";
import { Bot, User, Wrench } from "lucide-react";
import { useMemo } from "react";
import { parseTranscript, type TranscriptItem } from "../lib/transcript";

/**
 * Renders an OpenAI-format transcript. Every string is a React text child, so markup in customer or model text
 * stays inert; there is no markdown rendering and no HTML injection anywhere on this path.
 */
export function Transcript({ messages }: { messages: unknown }) {
  const items = useMemo(() => parseTranscript(messages), [messages]);
  if (!items.length) return <p className="muted text-[14px]">This row has no transcript.</p>;
  return (
    <ol className="grid gap-3" aria-label="Transcript">
      {items.map((it, i) => (
        <li key={i}>
          <Item item={it} />
        </li>
      ))}
    </ol>
  );
}

function Item({ item }: { item: TranscriptItem }) {
  if (item.kind === "message") {
    const user = item.role === "user";
    const Icon = user ? User : Bot;
    return (
      <div data-kind="message" data-role={item.role} className={clsx("rounded-lg border px-3.5 py-2.5", user ? "border-line bg-sunken" : "border-line bg-surface")}>
        <div className="eyebrow mb-1 flex items-center gap-1.5">
          <Icon size={13} aria-hidden="true" />
          {item.role}
        </div>
        <p className="pre-text m-0 text-[14px]">{item.text}</p>
      </div>
    );
  }
  if (item.kind === "tool") {
    return (
      <details open data-kind="tool" className="rounded-lg border border-accent bg-accent-soft/40 px-3.5 py-2.5">
        <summary className="flex cursor-pointer items-center gap-2 text-[13px] font-semibold">
          <Wrench size={14} aria-hidden="true" />
          <span className="eyebrow">Tool call</span>
          <code className="text-[13px]">{item.name}</code>
          {item.result === null ? <span className="chip chip-warn">no result</span> : null}
        </summary>
        {item.args ? (
          <div className="mt-2">
            <div className="eyebrow mb-1">Arguments</div>
            <pre data-kind="tool-args" className="pre-text m-0 rounded-md bg-surface px-3 py-2 text-[12.5px]">{item.args}</pre>
          </div>
        ) : null}
        {item.result !== null ? (
          <div className="mt-2">
            <div className="eyebrow mb-1">Tool result</div>
            <pre data-kind="tool-result" className="pre-text m-0 rounded-md border-l-4 border-pass bg-surface px-3 py-2 text-[12.5px]">{item.result}</pre>
          </div>
        ) : null}
      </details>
    );
  }
  return (
    <pre data-kind="unknown" className="pre-text m-0 rounded-lg border border-dashed border-line px-3 py-2 text-[12.5px]">{item.text}</pre>
  );
}
