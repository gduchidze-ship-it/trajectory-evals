import { DiffEditor, Editor, loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/editor/editor.worker?worker";
import jsonWorker from "monaco-editor/language/json/json.worker?worker";
import { useEffect, useState } from "react";

// Monaco loads from the local package, never a CDN: the tool must work offline.
self.MonacoEnvironment = {
  getWorker: (_id: string, label: string) => (label === "json" ? new jsonWorker() : new editorWorker()),
};
loader.config({ monaco });

monaco.editor.defineTheme("te-light", {
  base: "vs",
  inherit: true,
  rules: [],
  colors: { "editor.background": "#ffffff", "editorLineNumber.foreground": "#5a6870", "editorLineNumber.activeForeground": "#11181b" },
});
monaco.editor.defineTheme("te-dark", {
  base: "vs-dark",
  inherit: true,
  rules: [],
  colors: { "editor.background": "#0f1517", "editorLineNumber.foreground": "#8fa0a7", "editorLineNumber.activeForeground": "#e8eef0" },
});

function useMonacoTheme(): string {
  const read = () => {
    const pinned = document.documentElement.dataset["theme"];
    if (pinned === "dark" || pinned === "light") return pinned === "dark" ? "te-dark" : "te-light";
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "te-dark" : "te-light";
  };
  const [theme, setTheme] = useState(read);
  useEffect(() => {
    const update = () => setTheme(read());
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", update);
    return () => {
      obs.disconnect();
      mq.removeEventListener("change", update);
    };
  }, []);
  return theme;
}

export interface CodeEditorProps {
  value: string;
  onChange?: (v: string) => void;
  language: "ini" | "markdown" | "json" | "plaintext";
  readOnly?: boolean;
  height?: number | string;
  label: string;
  path?: string;
}

const OPTIONS: monaco.editor.IStandaloneEditorConstructionOptions = {
  minimap: { enabled: false },
  fontFamily: '"IBM Plex Mono", ui-monospace, Menlo, monospace',
  fontSize: 13,
  scrollBeyondLastLine: false,
  wordWrap: "on",
  tabSize: 2,
  renderLineHighlight: "line",
  padding: { top: 10, bottom: 10 },
  automaticLayout: true,
};

export default function CodeEditorImpl({ value, onChange, language, readOnly, height = 420, label, path }: CodeEditorProps) {
  const theme = useMonacoTheme();
  return (
    <div className="overflow-hidden rounded-lg border border-line" data-testid="code-editor" data-label={label}>
      <Editor
        height={height}
        language={language}
        theme={theme}
        value={value}
        path={path}
        onChange={(v) => onChange?.(v ?? "")}
        options={{ ...OPTIONS, readOnly: readOnly ?? false, ariaLabel: label }}
      />
    </div>
  );
}

export function CodeDiffImpl({ original, modified, height = 320, label }: { original: string; modified: string; height?: number | string; label: string }) {
  const theme = useMonacoTheme();
  return (
    <div className="overflow-hidden rounded-lg border border-line" data-testid="code-diff" data-label={label}>
      <DiffEditor
        height={height}
        language="markdown"
        theme={theme}
        original={original}
        modified={modified}
        originalModelPath={`diff/${label}/original`}
        modifiedModelPath={`diff/${label}/modified`}
        keepCurrentOriginalModel
        keepCurrentModifiedModel
        options={{ ...OPTIONS, readOnly: true, renderSideBySide: false, ariaLabel: label }}
      />
    </div>
  );
}
