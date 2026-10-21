import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type RowQuery } from "./client";
import type { DatasetKind, DatasetWrite, ResultKind, RunRequest } from "./types";

export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: ({ signal }) => api.health(signal), staleTime: 60_000 });

export const useAgents = () => useQuery({ queryKey: ["agents"], queryFn: api.agents });
export const useAgent = (id: string) => useQuery({ queryKey: ["agents", id], queryFn: () => api.agent(id) });

export const useCheck = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: { id: string; ping: boolean }) => api.check(v.id, v.ping),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["agents"] }),
  });
};

export const useAgentFile = (id: string, path: string | null) =>
  useQuery({ queryKey: ["agents", id, "files", path], queryFn: () => api.readFile(id, path ?? ""), enabled: path !== null });

export const useSaveFile = (id: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: { path: string; text: string }) => api.writeFile(id, v.path, v.text, false),
    onSuccess: (_d, v) => {
      void qc.invalidateQueries({ queryKey: ["agents"] });
      void qc.invalidateQueries({ queryKey: ["agents", id, "files", v.path] });
    },
  });
};

export const useDatasets = (agent?: string) =>
  useQuery({ queryKey: ["datasets", { agent: agent ?? null }], queryFn: () => api.datasets(agent) });

export const useDatasetPage = (path: string, offset: number, limit: number, kind?: DatasetKind) =>
  useQuery({
    queryKey: ["datasets", path, { offset, limit, kind: kind ?? null }],
    queryFn: () => api.dataset(path, { offset, limit, kind }),
    placeholderData: keepPreviousData,
  });

export const useSaveDataset = (path: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: { body: DatasetWrite; create?: boolean }) => api.writeDataset(path, v.body, { create: v.create }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["datasets"] }),
  });
};

export const useStartRun = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (b: RunRequest) => api.startRun(b),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["runs"] }),
  });
};

export const useRuns = () => useQuery({ queryKey: ["runs"], queryFn: api.runs, refetchInterval: 5000 });

export const useCancelRun = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.cancelRun(id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["runs"] }),
  });
};

export const useResults = (q: { agent?: string; kind?: ResultKind | ""; offset?: number; limit?: number }) =>
  useQuery({ queryKey: ["results", q], queryFn: () => api.results(q), placeholderData: keepPreviousData });

export const useResult = (id: string) => useQuery({ queryKey: ["results", id], queryFn: () => api.result(id) });
export const useEvalDetail = (id: string, evalId: string | undefined) =>
  useQuery({ queryKey: ["results", id, "evals", evalId], queryFn: () => api.evalDetail(id, evalId ?? ""), enabled: !!evalId });

export const useRows = (id: string, q: RowQuery, evalId?: string) =>
  useQuery({ queryKey: ["results", id, evalId ?? null, "rows", q], queryFn: () => api.rows(id, q, evalId), placeholderData: keepPreviousData });

export const useRow = (id: string, index: number | null, evalId?: string) =>
  useQuery({
    queryKey: ["results", id, evalId ?? null, "rows", index],
    queryFn: () => api.row(id, index ?? 0, evalId),
    enabled: index !== null,
  });

export const usePrompts = (id: string, candidate: string | null) =>
  useQuery({ queryKey: ["results", id, "prompts", candidate], queryFn: () => api.prompts(id, candidate ?? ""), enabled: candidate !== null });

export const useCompare = (a: string | null, b: string | null) =>
  useQuery({ queryKey: ["compare", a, b], queryFn: () => api.compare(a ?? "", b ?? ""), enabled: !!a && !!b });
