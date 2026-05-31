import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for Production E2E tests.
 *
 * These tests run against the built frontend served by the backend on port 8000.
 * This provides confidence that the production build + single-app serving works correctly.
 *
 * Usage:
 *   npm run test:e2e:prod
 *
 * Prerequisites:
 *   - Backend + built frontend must be running on http://localhost:8000
 *     (e.g. via `cd .. && ./run.ps1` or `run.bat`)
 */
export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',

  use: {
    baseURL: 'http://localhost:8000',
    trace: 'on-first-retry',
    // Grant mic permission so getUserMedia succeeds with the fake device flags below
    permissions: ['microphone'],
  },

  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // Enable fake media devices so getUserMedia succeeds in headless without real mic
        launchOptions: {
          args: [
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream',
          ],
        },
      },
    },
  ],

  // No webServer here — the production app must be started manually
  // (or via the project's run scripts) before running these tests.
});