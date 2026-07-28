import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development the viewer runs on Vite's port while the pipeline's data
// is served by `pos serve` on 8000. Proxying keeps the browser on one origin so
// there is no CORS story to get wrong, and the production build -- served by
// FastAPI itself out of viewer/dist -- uses the exact same relative paths.
// Where `pos serve` is listening. Change here if you use a different --port.
const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API, changeOrigin: true },
      "/frames": { target: API, changeOrigin: true },
      "/stream": { target: API, changeOrigin: true, ws: false },
    },
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
});
