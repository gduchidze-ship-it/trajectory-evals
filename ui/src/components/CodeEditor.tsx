import { lazy, Suspense } from "react";
import type { CodeEditorProps } from "./monacoImpl";
import { Skeleton } from "./ui";

const Impl = lazy(() => import("./monacoImpl"));
const DiffImpl = lazy(() => import("./monacoImpl").then((m) => ({ default: m.CodeDiffImpl })));

export function CodeEditor(props: CodeEditorProps) {
  return (
    <Suspense fallback={<div style={{ height: props.height ?? 420 }}><Skeleton className="h-full w-full" /></div>}>
      <Impl {...props} />
    </Suspense>
  );
}

export function CodeDiff(props: { original: string; modified: string; height?: number | string; label: string }) {
  return (
    <Suspense fallback={<div style={{ height: props.height ?? 320 }}><Skeleton className="h-full w-full" /></div>}>
      <DiffImpl {...props} />
    </Suspense>
  );
}
