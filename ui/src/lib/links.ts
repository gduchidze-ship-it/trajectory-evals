const seg = (p: string): string => p.split("/").map(encodeURIComponent).join("/");

export const agentHref = (id: string, tab?: string): string => `/agents/${encodeURIComponent(id)}${tab ? `?tab=${tab}` : ""}`;
export const datasetHref = (path: string, agent?: string | null): string =>
  `/datasets/${seg(path)}${agent ? `?agent=${encodeURIComponent(agent)}` : ""}`;
export const resultHref = (runId: string, evalId?: string): string =>
  `/results/${encodeURIComponent(runId)}${evalId ? `/evals/${encodeURIComponent(evalId)}` : ""}`;
export const jobHref = (id: string): string => `/runs/${id}`;
export const launchHref = (q: Record<string, string>): string => `/launch?${new URLSearchParams(q).toString()}`;
export const compareHref = (a: string, b: string): string => `/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`;
