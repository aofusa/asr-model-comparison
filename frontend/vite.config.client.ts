import { defineConfig } from 'vite';
import { qwikVite } from '@builder.io/qwik/optimizer';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig(() => {
  return {
    plugins: [
      qwikVite(),
      tsconfigPaths(),
    ],
    build: {
      rollupOptions: {
        input: 'src/entry.client.tsx',
      },
      outDir: 'dist',
      emptyOutDir: false,
    },
  };
});
