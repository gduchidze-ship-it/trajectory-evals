import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { makeQueryClient, routes } from "./App";
import "./styles.css";

async function boot(): Promise<void> {
  if (import.meta.env.MODE === "mock") {
    const { startMocking } = await import("./mocks/browser");
    await startMocking();
  }
  const el = document.getElementById("root");
  if (!el) throw new Error("#root is missing from index.html");
  const client = makeQueryClient();
  const router = createBrowserRouter(routes);
  createRoot(el).render(
    <StrictMode>
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </StrictMode>,
  );
}

void boot();
