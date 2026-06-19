import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    assetsDir: "webui-assets",
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/assets": "http://127.0.0.1:8765",
    },
  },
});
