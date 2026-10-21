import { expect, test } from "@playwright/test";
import { tabTo } from "./helpers";

test.describe("run launcher and live run", () => {
  test("launch an eval, watch it live, open the results", async ({ page }) => {
    await page.goto("/launch");
    await page.getByLabel("Agent", { exact: true }).selectOption("examples.support_agent");
    await page.getByLabel("Golden dataset", { exact: true }).selectOption("examples/support_agent/datasets/golden.sample.jsonl");
    await page.getByRole("button", { name: "Launch eval" }).click();

    await expect(page).toHaveURL(/\/runs\/[0-9a-f]{12}$/);
    await expect(page.getByRole("progressbar", { name: /Rows finished/ })).toBeVisible();
    await expect(page.getByRole("link", { name: "Open results" })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole("log", { name: "Job log" })).toContainText("started support_agent-eval");
    await expect(page.getByRole("table", { name: /Finished rows/ }).getByText("refund-eligible")).toBeVisible();

    await page.getByRole("link", { name: "Open results" }).click();
    await expect(page.getByText("Pass rate", { exact: true }).first()).toBeVisible();
  });

  test("cancel a running job and see the active-jobs badge", async ({ page }) => {
    await page.goto("/runs/7a1c3e5f9b02");
    await expect(page.getByTestId("active-jobs")).toBeVisible();
    await expect(page.getByRole("button", { name: "Cancel" })).toBeEnabled();
    page.once("dialog", (d) => void d.accept());
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(page.getByText(/Nothing was written for this job/)).toBeVisible();
    await expect(page.getByRole("button", { name: "Cancel" })).toBeDisabled();
  });

  test("an unknown job is explained, not blank", async ({ page }) => {
    await page.goto("/runs/ffffffffffff");
    await expect(page.getByText("The server does not know this job")).toBeVisible();
  });

  test("the launcher works with the keyboard alone", async ({ page }) => {
    await page.goto("/launch");
    await expect(page.getByRole("button", { name: "Launch eval" })).toBeVisible();

    // The radio group moves with the arrow keys.
    await tabTo(page, 'input[name="command"]:checked');
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("radio", { name: /Simulate/ })).toBeChecked();
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByRole("radio", { name: /Eval/ })).toBeChecked();

    // Submitting nothing explains what is missing and moves focus to the first bad field.
    await tabTo(page, "button[type=submit]");
    await page.keyboard.press("Enter");
    await expect(page.getByRole("alert")).toContainText("Fix these 2 fields");
    await expect(page.locator("#f-agent")).toBeFocused();

    await page.keyboard.type("support");
    await expect(page.locator("#f-agent")).toHaveValue("examples.support_agent");
    await expect(page.locator("#f-data")).toBeEnabled();
    await page.keyboard.press("Tab");
    await expect(page.locator("#f-data")).toBeFocused();
    await page.keyboard.type("golden");
    await expect(page.locator("#f-data")).toHaveValue(/golden\.sample\.jsonl$/);

    await tabTo(page, "button[type=submit]");
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/runs\/[0-9a-f]{12}$/);
    await expect(page.getByRole("link", { name: "Open results" })).toBeVisible({ timeout: 15_000 });

    // Global go-to shortcuts.
    await page.locator("body").press("g");
    await page.locator("body").press("r");
    await expect(page).toHaveURL(/\/results$/);
  });
});
