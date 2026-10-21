import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Link, Navigate, Route, Routes } from "react-router-dom";
import { ApiError } from "./api/client";
import { Shell } from "./components/Shell";
import { EmptyState, Page } from "./components/ui";
import { AgentDetailPage } from "./pages/AgentDetail";
import { AgentsPage } from "./pages/Agents";
import { ComparePage } from "./pages/Compare";
import { DatasetEditorPage } from "./pages/DatasetEditor";
import { JobsPage } from "./pages/Jobs";
import { LangSmithPage } from "./pages/LangSmith";
import { LauncherPage } from "./pages/Launcher";
import { ResultDetailPage } from "./pages/ResultDetail";
import { ResultsPage } from "./pages/Results";
import { RunLivePage } from "./pages/RunLive";

export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        staleTime: 5_000,
        retry: (count, error) => !(error instanceof ApiError && error.status < 500) && count < 1,
      },
    },
  });
}

function NotFound() {
  return (
    <Page title="Page not found">
      <EmptyState title="There is nothing at this address" action={<Link className="btn btn-primary" to="/agents">Go to agents</Link>} />
    </Page>
  );
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<Navigate to="/agents" replace />} />
        <Route path="agents" element={<AgentsPage />} />
        <Route path="agents/:agentId" element={<AgentDetailPage />} />
        <Route path="datasets/*" element={<DatasetEditorPage />} />
        <Route path="launch" element={<LauncherPage />} />
        <Route path="runs" element={<JobsPage />} />
        <Route path="runs/:jobId" element={<RunLivePage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="results/:runId" element={<ResultDetailPage />} />
        <Route path="results/:runId/evals/:evalId" element={<ResultDetailPage />} />
        <Route path="compare" element={<ComparePage />} />
        <Route path="langsmith" element={<LangSmithPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}

export function App({ client }: { client: QueryClient }) {
  return (
    <QueryClientProvider client={client}>
      <AppRoutes />
    </QueryClientProvider>
  );
}
