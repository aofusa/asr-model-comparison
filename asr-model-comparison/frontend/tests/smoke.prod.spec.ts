import { test, expect } from '@playwright/test';

/**
 * Production Smoke Tests
 *
 * These tests are intended to run against the built frontend + backend
 * (single-app mode on http://localhost:8000).
 *
 * Run with:
 *   npm run test:e2e:prod
 *
 * Prerequisites:
 *   The full application must be running in production mode:
 *   - Frontend built and served by the backend
 *   - Backend running on port 8000
 */

test.describe('Production Smoke Tests', () => {
  test('app loads from production build', async ({ page }) => {
    await page.goto('/');

    // Basic smoke checks that the built assets are served correctly.
    // Using more specific selectors to work reliably with the static shell
    // injected for backend serving (avoids duplicate text strict mode issues).
    await expect(page.getByText('ASR Real-time Comparison')).toBeVisible();
    await expect(page.getByTestId('model-label')).toBeVisible();
    await expect(page.getByRole('button', { name: /Start Recording/i })).toBeVisible();
  });

  test('settings panel is functional in built version', async ({ page }) => {
    await page.goto('/');

    const panel = page.locator('.settings-panel');
    await expect(panel).toBeVisible();

    // These match the static shell content injected for reliable backend serving
    await expect(panel.getByText('Beam Size')).toBeVisible();
    await expect(panel.getByText('Temperature')).toBeVisible();
    await expect(panel.getByText('Use Dedicated Class')).toBeVisible();
  });

  test('volume meter element exists in production build', async ({ page }) => {
    await page.goto('/');

    // The volume meter should be present (even if not recording yet)
    const meter = page.getByTestId('volume-meter');
    await expect(meter).toBeVisible();
  });
});