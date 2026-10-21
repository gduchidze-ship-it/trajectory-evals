import { expect, test } from "@playwright/test";
import { setHealth } from "./helpers";

test.describe("LangSmith", () => {
  test("harvest as a job, review the rows, approve one, then push is still refused", async ({ page }) => {
    await page.goto("/langsmith");
    await page.getByLabel("Agent", { exact: true }).selectOption("examples.sales_assistant");
    await page.getByLabel("Only errored runs").check();
    await page.getByRole("button", { name: "Start harvest" }).click();
    await expect(page).toHaveURL(/\/runs\/[0-9a-f]{12}$/);
    await expect(page.getByRole("link", { name: "Review them" })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/Wrote 6 rows/)).toBeVisible();

    await page.getByRole("link", { name: "Open the review queue" }).click();
    await expect(page.getByRole("table", { name: /awaiting review/ })).toContainText("4");
    await page.getByRole("link", { name: /Review 4 rows/ }).click();

    await expect(page.getByText(/Review mode: 4 rows/)).toBeVisible();
    await page.getByRole("row", { name: /run-0003/ }).click();
    await page.getByLabel(/Expected output/).fill("Saturday at 10:00 works.");
    await page.getByRole("button", { name: "Approve row" }).click();
    await expect(page.getByText(/Review mode: 3 rows/)).toBeVisible();
    await page.getByRole("button", { name: "Save file" }).click();
    await expect(page.getByText("Saved.")).toBeVisible();

    await page.goto("/langsmith?tab=push");
    await page.getByLabel("Agent", { exact: true }).selectOption("examples.sales_assistant");
    await page.getByLabel("Golden dataset").selectOption("data/langsmith/sales_assistant-20261021_120000.jsonl");
    await page.getByRole("button", { name: "Push dataset" }).click();
    await expect(page.getByText(/still have review_required set/)).toBeVisible();
    await expect(page.getByRole("link", { name: "Open the review queue" })).toBeVisible();

    await page.getByLabel("Golden dataset").selectOption("examples/sales_assistant/datasets/golden.jsonl");
    await page.getByRole("button", { name: "Push dataset" }).click();
    await expect(page.getByText("Pushed examples/sales_assistant/datasets/golden.jsonl as dataset sales_assistant/golden.")).toBeVisible();
  });

  test("score is a dry run by default and shows its key counts", async ({ page }) => {
    await page.goto("/langsmith?tab=score");
    await page.getByLabel("Agent", { exact: true }).selectOption("examples.sales_assistant");
    await expect(page.getByLabel(/Dry run/)).toBeChecked();
    await page.getByRole("button", { name: "Start scoring" }).click();
    const table = page.getByRole("table", { name: "Counts per feedback key" });
    await expect(table).toBeVisible({ timeout: 15_000 });
    await expect(table).toContainText("helpfulness");
  });

  test("pull reports an existing file", async ({ page }) => {
    await page.goto("/langsmith?tab=pull");
    await page.getByLabel("Agent", { exact: true }).selectOption("examples.sales_assistant");
    await page.getByLabel("LangSmith dataset name").fill("sales-golden");
    await page.getByLabel("Output file").fill("data/langsmith/sales_assistant-20261021_120000.jsonl");
    await page.getByRole("button", { name: "Pull dataset" }).click();
    await expect(page.getByText("the output file already exists")).toBeVisible();
  });

  test("without a key and outside demo, actions are disabled with an explanation", async ({ page }) => {
    await setHealth(page, { demo: false, langsmith_key: false });
    await page.goto("/langsmith");
    await expect(page.getByText(/LangSmith actions are disabled/)).toBeVisible();
    await expect(page.getByText(/scripted models/)).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Start harvest" })).toBeDisabled();
    await page.getByRole("tab", { name: "Review queue" }).click();
    await expect(page.getByRole("table", { name: /awaiting review/ })).toBeVisible();

    await page.goto("/launch?command=optimize");
    await expect(page.getByLabel("Upload candidates as LangSmith experiments")).toBeDisabled();
    await expect(page.getByText(/LANGSMITH_API_KEY is not set, so runs stay local/).first()).toBeVisible();
  });
});
