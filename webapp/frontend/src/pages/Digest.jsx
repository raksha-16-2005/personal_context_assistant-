import { useEffect, useState } from 'react'
import { api } from '../api'

export default function Digest() {
  const [digest, setDigest] = useState(null)
  const [settings, setSettings] = useState(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api.digest().then(setDigest).catch(() => {})
    api.digestSettings().then(setSettings).catch(() => {})
  }, [])

  async function toggle() {
    setSaving(true)
    try {
      setSettings(await api.saveDigestSettings(!settings.enabled, settings.send_hour_utc))
    } finally {
      setSaving(false)
    }
  }

  async function changeHour(e) {
    setSettings(await api.saveDigestSettings(settings.enabled, Number(e.target.value)))
  }

  return (
    <div className="page">
      <h1>Daily digest</h1>
      <p className="hint">
        Generated in-app on a schedule, not emailed - see the mailbox icon on this
        page whenever it's ready, rather than a new inbox message every day.
      </p>

      {settings && (
        <div className="digest-settings">
          <label>
            <input
              type="checkbox"
              checked={settings.enabled}
              disabled={saving}
              onChange={toggle}
            />
            Generate a daily digest
          </label>
          <label>
            at UTC hour
            <select
              value={settings.send_hour_utc}
              onChange={changeHour}
              disabled={!settings.enabled}
            >
              {Array.from({ length: 24 }, (_, h) => (
                <option key={h} value={h}>
                  {String(h).padStart(2, '0')}:00
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

      {digest?.id ? (
        <div className="digest-content">
          <p className="digest-meta">
            Generated {new Date(digest.created_at).toLocaleString()}
          </p>
          <section>
            <h2>Due soon</h2>
            {digest.due_soon.length === 0 && (
              <p className="empty-state">Nothing due in the next week.</p>
            )}
            <ul>
              {digest.due_soon.map((c, i) => (
                <li key={i}>
                  {c.due_at} - {c.text}
                  {c.counterparty && ` (with ${c.counterparty})`}
                </li>
              ))}
            </ul>
          </section>
          <p>{digest.pending_calendar_suggestions} pending calendar suggestion(s).</p>
          <p>{digest.new_messages} new message(s) since the last digest.</p>
        </div>
      ) : (
        <p className="empty-state">
          No digest generated yet - turn it on above and it will appear here at
          your chosen hour.
        </p>
      )}
    </div>
  )
}
