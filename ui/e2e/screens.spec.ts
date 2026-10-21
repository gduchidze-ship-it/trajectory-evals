import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { E1, O1, R1, S1, waitForMonaco } from "./helpers";

interface Screen {
  name: string;
  path: string;
  ready: (p: Page) => Promise<void>;
}

const screens: Screen[] = [
  { name: "agents", path: "/agents", ready: (p) => p.getByText("support_agent").first().waitFor() },
  { name: "agent-files", path: "/agents/examples.support_agent?tab=files", ready: waitForMonaco },
  { name: "dataset", path: "/datasets/examples/support_agent/datasets/golden.sample.jsonl", ready: (p) => p.getByText("refund-eligible").first().waitFor() },
  { name: "launcher", path: "/launch?agent=examples.support_agent", ready: (p) => p.getByRole("button", { name: /Launch eval/ }).waitFor() },
  { name: "live-run", path: "/runs/7a1c3e5f9b02", ready: async (p) => { await p.getByRole("table", { name: /Finished rows/ }).waitFor(); await p.waitForTimeout(900); } },
  { name: "results", path: "/results", ready: (p) => p.getByRole("table", { name: "Runs" }).waitFor() },
  { name: "result-eval", path: `/results/${E1}?row=2`, ready: (p) => p.getByRole("dialog").getByText("Transcript").waitFor() },
  { name: "result-simulate", path: `/results/${S1}`, ready: (p) => p.getByRole("img", { name: "Attack success rate per persona" }).waitFor() },
  { name: "result-redteam", path: `/results/${R1}`, ready: (p) => p.getByText("The attack success rate is uncertain").waitFor() },
  { name: "compare", path: "/compare?a=20261021_101200_support-agent-eval-local_3b7e10&b=20261021_121530_support-agent-eval-local_a91c2d", ready: (p) => p.getByRole("region", { name: "Fixed" }).waitFor() },
  { name: "optimize", path: `/results/${O1}`, ready: async (p) => { await waitForMonaco(p); } },
  { name: "langsmith", path: "/langsmith", ready: (p) => p.getByRole("button", { name: "Start harvest" }).waitFor() },
];

for (const scheme of ["light", "dark"] as const) {
  test.describe(`screens in ${scheme} mode`, () => {
    test.use({ colorScheme: scheme });
    for (const s of screens) {
      test(`${s.name}`, async ({ page }) => {
        await page.goto(s.path);
        await s.ready(page);
        await page.waitForTimeout(400);
        await page.screenshot({ path: `test-results/screens/${s.name}-${scheme}.png`, fullPage: s.name !== "result-eval" });
        const axe = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).exclude(".monaco-editor").exclude(".monaco-diff-editor").analyze();
        const bad = axe.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
        expect(bad.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`)).toEqual([]);
      });
    }
  });
}

test("a pinned theme survives a reload and beats the OS setting", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/agents");
  await page.getByRole("button", { name: /Theme: system/ }).click();
  await page.getByRole("button", { name: /Theme: light/ }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
});
