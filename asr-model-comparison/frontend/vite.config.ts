import { defineConfig } from 'vite';
import { qwikVite } from '@builder.io/qwik/optimizer';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig(() => {
  return {
    plugins: [qwikVite(), tsconfigPaths()],
    // Explicitly treat index.html as the client entry so that `vite build`
    // emits a proper index.html shell (required for SPA single-app mode).
    // This is necessary because we use pure qwikVite() without Qwik City.
    // Note: We intentionally do NOT set rollupOptions.input here.
    // With qwikVite() the HTML plugin + optimizer decide the client bootstrap.
    // We rely on <script src="/src/entry.dev.tsx"> in index.html for both dev and build.
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