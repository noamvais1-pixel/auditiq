import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxied so the browser only ever talks to one origin -- keeps CORS out of
    // the dev setup and matches how this would sit behind a reverse proxy.
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
