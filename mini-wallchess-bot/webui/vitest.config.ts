import { defineConfig } from 'vitest/config'

// Standalone config so unit tests don't load the app's Cloudflare/Start Vite
// plugins (which reject vitest's `resolve.external` env options).
export default defineConfig({
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
