import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";
const KEY = "te-theme";

function read(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

function apply(t: Theme): void {
  if (t === "system") delete document.documentElement.dataset["theme"];
  else document.documentElement.dataset["theme"] = t;
}

export function useTheme(): { theme: Theme; cycle: () => void; set: (t: Theme) => void } {
  const [theme, setTheme] = useState<Theme>(read);
  useEffect(() => apply(theme), [theme]);
  const set = useCallback((t: Theme) => {
    setTheme(t);
    try {
      if (t === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, t);
    } catch {
      // storage unavailable: the choice lasts for this tab only
    }
  }, []);
  const cycle = useCallback(() => set(theme === "system" ? "light" : theme === "light" ? "dark" : "system"), [theme, set]);
  return { theme, cycle, set };
}
