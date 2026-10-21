import { expect, test } from "@playwright/test";
import { waitForMonaco } from "./helpers";

test.describe("agents and datasets", () => {
  test("check an agent, edit a prompt with inline validation, and read a dataset's row errors", async ({ page }) => {
    await page.goto("/agents");
    await expect(page.getByText(/scripted models, nothing leaves this machine/)).toBeVisible();
    await expect(page.getByText("…/agent.toml: [graders] No module named 'src'")).toBeVisible();

    await page.getByRole("button", { name: "Check support_agent" }).click();
    await expect(page.getByText("Check passed")).toBeVisible();
    await expect(page.getByRole("list", { name: "Check steps" }).getByText("card")).toBeVisible();

    await page.getByRole("link", { name: "support_agent", exact: true }).click();
    await page.getByRole("tab", { name: /Files/ }).click();
    await page.getByRole("button", { name: /prompts\/system\.md/ }).click();
    await waitForMonaco(page);
    const editor = page.locator(".monaco-editor").first();
    await editor.click();
    await page.keyboard.press("ControlOrMeta+End");

    // A valid edit passes the debounced dry run and can be saved.
    await page.keyboard.type("\nAnswer in one short paragraph.");
    await expect(page.getByText("Valid", { exact: true })).toBeVisible();
    const save = page.getByRole("button", { name: "Save", exact: true });
    await expect(save).toBeEnabled();
    await save.click();
    await expect(page.getByText("Saved", { exact: true })).toBeVisible();

    // A broken config is refused inline and the save stays disabled.
    await page.getByRole("button", { name: "agent.toml" }).click();
    await page.locator(".monaco-editor:visible").first().click();
    await page.keyboard.press("ControlOrMeta+End");
    await page.keyboard.type("\nBROKEN = 1");
    await expect(page.getByRole("alert").filter({ hasText: "unknown key 'BROKEN'" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Save", exact: true })).toBeDisabled();

    // Leaving with an unsaved edit asks first; dismissing keeps the editor, accepting leaves.
    const [stay] = await Promise.all([page.waitForEvent("dialog"), page.getByRole("tab", { name: /Datasets/ }).click()]);
    expect(stay.message()).toContain("unsaved edits");
    await stay.dismiss();
    await expect(page.getByRole("button", { name: "Save", exact: true })).toBeVisible();
    const [leave] = await Promise.all([page.waitForEvent("dialog"), page.getByRole("tab", { name: /Datasets/ }).click()]);
    await leave.accept();
    await page.getByRole("link", { name: "golden.sample.jsonl" }).click();
    await expect(page.getByText("2 problems")).toHaveCount(0);
    await expect(page.getByRole("region", { name: "Row problems" })).toContainText("input: field required");
    await page.getByRole("button", { name: "Jump to line 8" }).click();
    await expect(page.getByRole("complementary", { name: "Row editor" })).toContainText("Line 8");
  });

  test("a read-only dataset says why", async ({ page }) => {
    await page.goto("/datasets/examples/sales_assistant/datasets/golden.jsonl");
    await expect(page.getByText(/Read-only: committed file/)).toBeVisible();
    await expect(page.getByRole("button", { name: "Save file" })).toBeDisabled();
  });
});
