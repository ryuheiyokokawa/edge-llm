import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@edge-llm/core": path.resolve(__dirname, "../../packages/core/src"),
      "@edge-llm/react": path.resolve(__dirname, "../../packages/react/src"),
    },
  },
  server: {
    port: 3000,
    open: true,
    headers: {
      // Required for SharedArrayBuffer which some WASM operations need
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  optimizeDeps: {
    // Exclude WebLLM from optimization (uses WebGPU, handled separately)
    exclude: ["@mlc-ai/web-llm"],
    // Include transformers.js deps for proper bundling
    include: ["onnxruntime-web"],
  },
  worker: {
    format: "es",
  },
});
