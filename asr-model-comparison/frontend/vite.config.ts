import { defineConfig } from 'vite';
import { qwikVite } from '@builder.io/qwik/optimizer';
import { qwikCity } from '@builder.io/qwik-city/vite';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig(() => {
  return {
    plugins: [
      // Qwik City must come before qwikVite()
      qwikCity(),
      qwikVite(),
      tsconfigPaths(),
    ],
    build: {
      // We must explicitly specify index.html as input.
      // Qwik City 1.20 + our custom root/routes setup does not auto-emit a
      // usable static index.html without this.
      rollupOptions: {
        input: 'index.html',
      },
    },
    // Qwik City + qwikVite() combination produces a proper static index.html
    // during `qwik build`, which is ideal for single-app mode (FastAPI serving
    // the built frontend from backend/static/).
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