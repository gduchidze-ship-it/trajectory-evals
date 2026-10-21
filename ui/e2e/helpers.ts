import type { Page } from "@playwright/test";

export const E1 = "20261021_114757_support-agent-eval-local_f0736f";
export const E0 = "20261021_101200_support-agent-eval-local_3b7e10";
export const E2 = "20261021_121530_support-agent-eval-local_a91c2d";
export const S1 = "20261021_113000_support-agent-simulate-local_c4d2e8";
export const R1 = "20261021_112200_support-agent-redteam-local_5e6f70";
export const O1 = "20261021_105500_support-agent-optimize-local_9a8b7c";
export const SW1 = "20261021_104000_support-agent-sweep-local_1d2e3f";

/** Tab until the element matching `selector` has focus; fails if it takes more than `max` presses. */
export async function tabTo(page: Page, selector: string, max = 60): Promise<void> {
  for (let i = 0; i < max; i++) {
    const hit = await page.evaluate((s) => document.activeElement?.matches(s) ?? false, selector);
    if (hit) return;
    await page.keyboard.press("Tab");
  }
  throw new Error(`Never reached ${selector} with Tab`);
}

export async function waitForMonaco(page: Page): Promise<void> {
  await page.locator(".monaco-editor:visible").first().waitFor({ state: "visible", timeout: 30_000 });
}

export async function setHealth(page: Page, health: { demo: boolean; langsmith_key: boolean }): Promise<void> {
  await page.addInitScript((h) => localStorage.setItem("te-mock", JSON.stringify(h)), health);
}
