import { useEffect, useState } from 'react'
import { api, ApiError } from '../api'

// Marketing mailers (Customer.io among them) generate their plain-text
// alternative by writing every link as "label ( url )" - genuinely present
// in the stored body (see enron.py's _body_of, which extracts text/plain
// unmodified because the retrieval pipeline needs it untouched), but a wall
// of tracking-link parentheticals is not what a person opening "view full
// email" wants to read. Stripped here, display-only - the fetched `body`
// value itself, and everything upstream of it, is untouched.
function cleanEmailBody(text) {
  return text.replace(/\s*\(\s*https?:\/\/[^\s)]+\s*\)/g, '').replace(/\n{3,}/g, '\n\n').trim()
}

function formatDate(iso) {
  if (!iso) return 'unknown'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

// Shared between Chat (a citation) and Commitments (a suggestion's source
// message) - both just need "the full email behind this message_id", fetched
// from the same GET /messages/{id} endpoint.
export default function EmailModal({ messageId, onClose }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setData(null)
    setError('')
    api.message(messageId)
      .then(setData)
      .catch((err) =>
        setError(err instanceof ApiError ? err.message : 'could not load this email'))
  }, [messageId])

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        {error && <p className="error">{error}</p>}
        {!data && !error && <p>Loading…</p>}
        {data && (
          <>
            <h3>{data.subject || '(no subject)'}</h3>
            <dl className="email-meta">
              <div><dt>From</dt><dd>{data.sender || 'unknown'}</dd></div>
              <div><dt>To</dt><dd>{data.recipients || 'unknown'}</dd></div>
              {data.cc && <div><dt>Cc</dt><dd>{data.cc}</dd></div>}
              <div><dt>Date</dt><dd>{formatDate(data.date)}</dd></div>
            </dl>
            <pre className="email-body">{data.body ? cleanEmailBody(data.body) : '(no content)'}</pre>
          </>
        )}
      </div>
    </div>
  )
}
