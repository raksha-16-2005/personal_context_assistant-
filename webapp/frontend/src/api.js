// Thin fetch wrapper. Paths are relative by default - the dev server proxies
// them to FastAPI (vite.config.js) and, when this app is served *by* FastAPI
// itself in production, there is no different origin to configure. Set
// VITE_API_BASE_URL only when the frontend is deployed separately from the
// backend (e.g. this app on Vercel, FastAPI on Fly).
export const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`)
    this.status = status
  }
}

async function request(path, options = {}) {
  const resp = await fetch(`${API_BASE}${path}`, {
    credentials: 'include',
    headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
    ...options,
  })
  if (resp.status === 204) return null

  const isJson = (resp.headers.get('content-type') || '').includes('application/json')
  const data = isJson ? await resp.json() : await resp.text()
  if (!resp.ok) {
    throw new ApiError(resp.status, isJson ? data.detail : data)
  }
  return data
}

export const api = {
  me: () => request('/me'),
  logout: () => request('/auth/google/logout', { method: 'POST' }),
  deleteAccount: () => request('/account', { method: 'DELETE' }),

  geminiKeyStatus: () => request('/account/gemini-key'),
  setGeminiKey: (apiKey) =>
    request('/account/gemini-key', { method: 'PUT', body: JSON.stringify({ api_key: apiKey }) }),
  deleteGeminiKey: () => request('/account/gemini-key', { method: 'DELETE' }),
  setTimezone: (timezone) =>
    request('/account/timezone', { method: 'PUT', body: JSON.stringify({ timezone }) }),

  conversations: () => request('/conversations'),
  conversation: (id) => request(`/conversations/${id}`),
  chat: (question, conversationId) =>
    request('/chat', {
      method: 'POST',
      body: JSON.stringify({ question, conversation_id: conversationId }),
    }),
  message: (messageId) => request(`/messages/${encodeURIComponent(messageId)}`),

  suggestions: (status = 'pending') =>
    request(`/calendar/suggestions?status=${encodeURIComponent(status)}`),
  confirmSuggestion: (id) => request(`/calendar/suggestions/${id}/confirm`, { method: 'POST' }),
  dismissSuggestion: (id) => request(`/calendar/suggestions/${id}/dismiss`, { method: 'POST' }),

  digest: () => request('/digest/latest'),
  digestSettings: () => request('/digest/settings'),
  saveDigestSettings: (enabled, sendHourUtc) =>
    request('/digest/settings', {
      method: 'PUT',
      body: JSON.stringify({ enabled, send_hour_utc: sendHourUtc }),
    }),
}
