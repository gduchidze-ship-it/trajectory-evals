import { QueryClient } from "@tanstack/react-query";
import { Link, Navigate, useRouteError, type RouteObject } from "react-router-dom";
import { ApiError } from "./api/client";
import { Shell } from "./components/Shell";
import { EmptyState, ErrorState, Page } from "./components/ui";
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

/** Anything a screen throws lands here instead of a blank page. */
function RouteError() {
  const error = useRouteError();
  return (
    <Page title="Something went wrong" back={<Link to="/agents">Agents</Link>}>
      <ErrorState error={error} title="This screen failed to render" />
    </Page>
  );
}

export const routes: RouteObject[] = [
  {
    element: <Shell />,
    children: [
      {
        // The error boundary sits below the shell, so a failing screen keeps the sidebar and navigation.
        errorElement: <RouteError />,
        children: [
      { index: true, element: <Navigate to="/agents" replace /> },
      { path: "agents", element: <AgentsPage /> },
      { path: "agents/:agentId", element: <AgentDetailPage /> },
      { path: "datasets/*", element: <DatasetEditorPage /> },
      { path: "launch", element: <LauncherPage /> },
      { path: "runs", element: <JobsPage /> },
      { path: "runs/:jobId", element: <RunLivePage /> },
      { path: "results", element: <ResultsPage /> },
      { path: "results/:runId", element: <ResultDetailPage /> },
      { path: "results/:runId/evals/:evalId", element: <ResultDetailPage /> },
      { path: "compare", element: <ComparePage /> },
      { path: "langsmith", element: <LangSmithPage /> },
      { path: "*", element: <NotFound /> },
        ],
      },
    ],
  },
];
