import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  // Real WebSocket protocol verification (ws-protocol.prod.spec.ts) is excluded from `npm run test:e2e` (dev).
  // Per 修正指示書: devでは実接続テストは不要 (UI/モック再接続/smoke中心で十分)。prod専用 (test:e2e:prod) で実行。
  // TDD記録: 変更前はdevでもこのテストが走り flake で1 fail していた。変更によりdevは安定フルパス。
  testIgnore: ['**/ws-protocol.prod.spec.ts'],
  use: {
    baseURL: 'http://localhost:5173',
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
  webServer: {
    // Use Vite directly (qwik dev CLI wrapper has environment-specific module resolution issues on some Windows setups).
    // The qwikVite() plugin in vite.config.ts still gives full Qwik dev experience (HMR, optimizer).
    // Note: --mode ssr (from package dev) can cause root JSX render errors in some setups; direct vite used for stability.
    command: 'npx vite --port 5173',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});
