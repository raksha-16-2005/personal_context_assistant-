import { useEffect, useState } from 'react'
import { api } from '../api'

// Every zone the browser's own Intl implementation knows about - no
// hardcoded list to keep in sync with the IANA database, and it's already
// what `zoneinfo` on the backend resolves against. Falls back to a short,
// common list on a browser old enough not to have `supportedValuesOf`.
const TIMEZONES = typeof Intl.supportedValuesOf === 'function'
  ? Intl.supportedValuesOf('timeZone')
  : ['UTC', 'America/New_York', 'America/Los_Angeles', 'Europe/London',
     'Europe/Berlin', 'Asia/Kolkata', 'Asia/Tokyo', 'Australia/Sydney']

export default function Settings() {
  const [hasKey, setHasKey] = useState(false)
  const [keyInput, setKeyInput] = useState('')
  const [saving, setSaving] = useState(false)
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [message, setMessage] = useState('')
  const [timezone, setTimezoneState] = useState('')
  const [savingTimezone, setSavingTimezone] = useState(false)

  useEffect(() => {
    api.geminiKeyStatus().then((r) => setHasKey(r.has_key)).catch(() => {})
    api.me().then((me) => setTimezoneState(me.timezone)).catch(() => {})
  }, [])

  async function changeTimezone(e) {
    const tz = e.target.value
    setSavingTimezone(true)
    try {
      await api.setTimezone(tz)
      setTimezoneState(tz)
    } finally {
      setSavingTimezone(false)
    }
  }

  async function saveKey(e) {
    e.preventDefault()
    const trimmed = keyInput.trim()
    if (!trimmed) return
    setSaving(true)
    try {
      await api.setGeminiKey(trimmed)
      setKeyInput('')
      setHasKey(true)
      setMessage('Key saved.')
    } finally {
      setSaving(false)
    }
  }

  async function removeKey() {
    await api.deleteGeminiKey()
    setHasKey(false)
    setMessage('Key removed.')
  }

  async function logout() {
    await api.logout()
    window.location.href = '/login'
  }

  async function deleteAccount() {
    await api.deleteAccount()
    window.location.href = '/login'
  }

  return (
    <div className="page">
      <h1>Settings</h1>

      <section className="settings-section">
        <h2>Gemini API key</h2>
        <p className="hint">
          Used only for your own chat requests, encrypted at rest, and never
          shown again once saved.
        </p>
        <p>Status: {hasKey ? 'a key is on file' : 'no key saved yet'}</p>
        <form onSubmit={saveKey} className="key-form">
          <input
            type="password"
            value={keyInput}
            onChange={(e) => setKeyInput(e.target.value)}
            placeholder="Paste your Gemini API key"
          />
          <button type="submit" disabled={saving}>
            Save
          </button>
        </form>
        {hasKey && (
          <button className="link-button" onClick={removeKey}>
            Remove key
          </button>
        )}
        {message && <p className="hint">{message}</p>}
      </section>

      <section className="settings-section">
        <h2>Timezone</h2>
        <p className="hint">
          Used to resolve "today" and "this week" in chat and your daily
          digest to your own day, not the server's.
        </p>
        {timezone && (
          <select value={timezone} onChange={changeTimezone} disabled={savingTimezone}>
            {TIMEZONES.map((tz) => (
              <option key={tz} value={tz}>{tz}</option>
            ))}
          </select>
        )}
      </section>

      <section className="settings-section">
        <h2>Session</h2>
        <button onClick={logout}>Log out</button>
      </section>

      <section className="settings-section danger">
        <h2>Delete account</h2>
        <p className="hint">
          Deletes your account, tokens, saved key, conversations, commitments,
          and mailbox index. This cannot be undone.
        </p>
        {!confirmingDelete ? (
          <button className="danger-button" onClick={() => setConfirmingDelete(true)}>
            Delete my account
          </button>
        ) : (
          <div className="confirm-row">
            <span>Are you sure?</span>
            <button className="danger-button" onClick={deleteAccount}>
              Yes, delete everything
            </button>
            <button onClick={() => setConfirmingDelete(false)}>Cancel</button>
          </div>
        )}
      </section>
    </div>
  )
}
