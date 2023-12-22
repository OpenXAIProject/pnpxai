import { defineConfig } from "vitest/config"
import react from "@vitejs/plugin-react"

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    open: false,
  },
  build: {
    outDir: "build",
    sourcemap: true,
    chunkSizeWarningLimit: 1000, // Increase the limit to 1000 KB
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "src/setupTests",
    mockReset: true,
  },
})
