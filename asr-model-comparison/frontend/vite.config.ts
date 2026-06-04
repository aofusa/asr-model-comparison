import { defineConfig } from 'vite';
import { qwikVite } from '@builder.io/qwik/optimizer';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig(() => {
  return {
    plugins: [
      // Pure Qwik + Vite (no Qwik City) for dev server stability on Windows + prod static SPA.
      // See AGENT.md, CLAUDE.md, FRONTEND_QWIK_STATIC_BUILD_PROD_HYDRATION_FIX_PLAN.md
      qwikVite(),
      tsconfigPaths(),
    ],
    build: {
      // index.html input for static SPA build (SSR HTML + shell injection + postbuild ensure for client script).
      // See FRONTEND_QWIK_STATIC_BUILD_PROD_HYDRATION_FIX_PLAN.md
      rollupOptions: {
        input: 'index.html',
      },
    },
    // qwikVite produces static index.html for single-app mode (FastAPI serving backend/static).
    preview: {
      headers: {
        'Cache-Control': 'public, max-age=600',
      },
    },
    server: {
      port: 5173,
    },
  };
});