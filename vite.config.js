// Vite config for Freebuff hosting detection and static preview builds.
// The live simulation is served by the Python backend (run.py); this config
// only lets Vite bundle the static frontend in web/ when hosting runs
// `npm run build`.
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: 'web',
  plugins: [react()],
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
  server: {
    port: 8777,
  },
});
