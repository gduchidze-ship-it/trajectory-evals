import { setupWorker } from "msw/browser";
import { mockState } from "./fixtures";
import { handlers } from "./handlers";

/** Playwright and manual testing can override Health through localStorage["te-mock"]. */
function applyOverrides(): void {
  try {
    const raw = localStorage.getItem("te-mock");
    if (!raw) return;
    const o: unknown = JSON.parse(raw);
    if (typeof o !== "object" || o === null) return;
    const { demo, langsmith_key } = o as Record<string, unknown>;
    if (typeof demo === "boolean") mockState.health.demo = demo;
    if (typeof langsmith_key === "boolean") mockState.health.langsmith_key = langsmith_key;
  } catch {
    // storage unavailable: keep the defaults
  }
}

export async function startMocking(): Promise<void> {
  applyOverrides();
  await setupWorker(...handlers).start({ onUnhandledFrame: "bypass", quiet: true });
}
