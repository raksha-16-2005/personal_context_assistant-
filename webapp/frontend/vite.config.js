import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxies API calls to the FastAPI backend during `npm run dev` so the app
// can be written with plain relative fetch() calls - the same paths work
// unchanged once FastAPI serves the built `dist/` itself in production (see
// ../app/main.py), so there is only one version of every request URL to
// keep in sync, not a dev one and a prod one.
const API_PREFIXES = [
  '/auth', '/me', '/account', '/chat', '/conversations', '/messages', '/calendar', '/digest',
  '/healthz',
]

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(API_PREFIXES.map((prefix) => [prefix, {
      target: 'http://localhost:8000',
      changeOrigin: true,
    }])),
  },
})
