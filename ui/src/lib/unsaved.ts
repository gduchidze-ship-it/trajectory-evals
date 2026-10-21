import { useEffect } from "react";
import { useBlocker } from "react-router-dom";

/**
 * While `dirty`, warns before the tab closes or reloads and asks before any in-app navigation (links, tab
 * changes, go-to shortcuts). Needs a data router, which main.tsx and the tests provide.
 */
export function useUnsavedGuard(dirty: boolean): void {
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      dirty && (currentLocation.pathname !== nextLocation.pathname || currentLocation.search !== nextLocation.search),
  );
  useEffect(() => {
    if (blocker.state !== "blocked") return;
    if (window.confirm("You have unsaved edits. Discard them?")) blocker.proceed();
    else blocker.reset();
  }, [blocker]);
  useEffect(() => {
    if (!dirty) return;
    const onUnload = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, [dirty]);
}
