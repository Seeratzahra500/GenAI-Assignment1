import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = { "/api": { target: process.env.API_TARGET || "http://localhost:8000", changeOrigin: true } };

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: api },
  preview: { port: 3000, proxy: api },
});
