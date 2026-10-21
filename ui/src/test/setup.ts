import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterAll, afterEach, beforeAll } from "vitest";
import { mockState } from "../mocks/fixtures";
import { resetMockState } from "../mocks/handlers";
import { mockConfig, resetJobs } from "../mocks/jobs";
import { server } from "../mocks/server";

const initialHealth = { ...mockState.health };

beforeAll(() => {
  mockConfig.rowDelayMs = 1;
  server.listen({ onUnhandledFrame: "error" });
});
afterEach(() => {
  cleanup();
  server.resetHandlers();
  resetJobs();
  resetMockState();
  Object.assign(mockState.health, initialHealth);
});
afterAll(() => server.close());
