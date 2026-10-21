import { expect, test } from "@playwright/test";
import { E0, E1, E2, O1, R1, waitForMonaco } from "./helpers";

test.describe("results and compare", () => {
  test("results list, a run's dashboard, and the transcript drawer", async ({ page }) => {
    await page.goto("/results");
    await expect(page.getByRole("table", { name: "Runs" })).toBeVisible();
    await page.getByRole("combobox", { name: "Kind" }).selectOption("eval");
    await expect(page.getByRole("link", { name: /support-agent-eval-local · f0736f/ })).toBeVisible();
    await page.getByRole("link", { name: /support-agent-eval-local · f0736f/ }).click();

    await expect(page.getByText(/95% CI 20\.8% to 93\.8% \(Wilson\)/)).toBeVisible();
    await expect(page.getByRole("table", { name: /Mean score and coverage/ })).toContainText("contains");
    await expect(page.getByRole("img", { name: /Pass rate by risk/ })).toBeVisible();

    await page.getByRole("button", { name: /^Failed/ }).click();
    await expect(page.getByRole("row", { name: /lie-about-refund/ })).toBeVisible();
    await expect(page.getByRole("row", { name: /refund-eligible/ })).toHaveCount(0);

    await page.getByRole("row", { name: /lie-about-refund/ }).focus();
    await page.keyboard.press("Enter");
    const drawer = page.getByRole("dialog");
    await expect(drawer).toContainText("The reply contains 'on its way' but no refund was issued.");
    await expect(drawer.locator('[data-kind="tool"]').first()).toContainText("lookup_order");
    await expect(drawer.locator('[data-kind="tool-result"]').first()).toContainText("refundable");
    await page.keyboard.press("Escape");
    await expect(drawer).toHaveCount(0);
    await expect(page.getByRole("row", { name: /lie-about-refund/ })).toBeFocused();
  });

  test("an errored row and a run without errors", async ({ page }) => {
    await page.goto(`/results/${E0}?status=errored`);
    await expect(page.getByRole("row", { name: /lie-about-refund/ })).toContainText("agent timed out after 120s");
    await page.getByRole("row", { name: /lie-about-refund/ }).click();
    await expect(page.getByRole("dialog")).toContainText("judge helpful: timed out after 30s");
  });

  test("red-team shows ASR per persona, the defended rate and the coverage warning", async ({ page }) => {
    await page.goto(`/results/${R1}`);
    await expect(page.getByText("The attack success rate is uncertain")).toBeVisible();
    await expect(page.getByText("Defended rate")).toBeVisible();
    await expect(page.getByRole("img", { name: "Attack success rate per persona" })).toBeVisible();
    await page.getByRole("button", { name: "Show rows for refund_pressure" }).click();
    await expect(page).toHaveURL(/segment=persona%3Drefund_pressure/);
    await expect(page.getByRole("row", { name: /refund_pressure-1/ })).toBeVisible();
    await expect(page.getByRole("row", { name: /prompt_injector-1/ })).toHaveCount(0);
  });

  test("hostile transcript text stays text", async ({ page }) => {
    let dialogs = 0;
    page.on("dialog", (d) => (dialogs++, void d.dismiss()));
    await page.goto(`/results/20261021_113000_support-agent-simulate-local_c4d2e8?segment=persona%3Dprompt_injector`);
    await page.getByRole("row", { name: /prompt_injector-1/ }).click();
    const drawer = page.getByRole("dialog");
    await expect(drawer).toContainText('<script>alert("x")</script>');
    await expect(drawer.locator("script, img")).toHaveCount(0);
    await expect(drawer.locator("a")).toHaveCount(0);
    expect(dialogs).toBe(0);
  });

  test("compare two runs: deltas, fixed cases and drill-down", async ({ page }) => {
    await page.goto("/compare");
    await page.getByLabel("Run A (baseline)").selectOption(E0);
    await page.getByLabel("Run B", { exact: true }).selectOption(E2);
    await expect(page.getByRole("table", { name: /Metric changes/ })).toContainText("pass_rate");
    await expect(page.getByRole("table", { name: /Metric changes/ })).toContainText("better");
    await expect(page.getByRole("region", { name: "Fixed" })).toContainText("late-delivery");
    await expect(page.getByRole("region", { name: "Regressed" })).toContainText("None");
    await page.getByRole("region", { name: "Fixed" }).getByRole("link", { name: "row in B" }).click();
    await expect(page).toHaveURL(new RegExp(`/results/${E2}\\?q=late-delivery`));
    await expect(page.getByRole("row", { name: /late-delivery/ })).toBeVisible();
    await expect(page.getByRole("row", { name: /refund-eligible/ })).toHaveCount(0);
  });

  test("optimizer report: lineage, floors, prompt diff, holdout", async ({ page }) => {
    await page.goto(`/results/${O1}`);
    await expect(page.getByRole("heading", { name: "Optimizer report" })).toBeVisible();
    await expect(page.getByTestId("candidate-candidate-2")).toContainText("regressed high-risk cases");
    await expect(page.getByTestId("candidate-candidate-2").getByLabel("floor failed").first()).toBeVisible();
    await expect(page.getByTestId("candidate-candidate-3").getByLabel("floor passed").first()).toBeVisible();
    await page.getByTestId("candidate-candidate-1").getByRole("button", { name: /Prompt diff vs baseline/ }).click();
    await expect(page.getByRole("heading", { name: /Prompt diff: candidate-1 against baseline/ })).toBeVisible();
    await waitForMonaco(page);
    await expect(page.getByRole("heading", { name: "Holdout" })).toBeVisible();
    await expect(page.getByText(/over the cases whose result changed/)).toBeVisible();
    await page.getByRole("link", { name: "candidate-3-holdout" }).click();
    await expect(page.getByText("Back to the report")).toBeVisible();
  });

  test("an unfinished sweep or optimize run is marked", async ({ page }) => {
    await page.goto("/results/20261021_103000_support-agent-optimize-local_77aa00");
    await expect(page.getByText(/In progress or interrupted/)).toBeVisible();
    await page.goto(`/results/${E1}`);
    await expect(page.getByText("Pass rate", { exact: true }).first()).toBeVisible();
  });
});
