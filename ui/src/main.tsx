import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App, makeQueryClient } from "./App";
import "./styles.css";

async function boot(): Promise<void> {
  if (import.meta.env.MODE === "mock") {
    const { startMocking } = await import("./mocks/browser");
    await startMocking();
  }
  const el = document.getElementById("root");
  if (!el) throw new Error("#root is missing from index.html");
  createRoot(el).render(
    <StrictMode>
      <BrowserRouter>
        <App client={makeQueryClient()} />
      </BrowserRouter>
    </StrictMode>,
  );
}

void boot();
