import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev mode: Vite serves the frontend on :5173 and proxies /api
// (HTTP + WebSocket) to the Python backend on :8765.
// Production: `npm run build` emits dist/, which the FastAPI app
// serves itself — one process, one port.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
