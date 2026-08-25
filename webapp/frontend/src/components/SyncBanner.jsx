import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

const POLL_MS = 4000

function etaMessage(etaSeconds, messagesSeen) {
  const minutes = Math.max(1, Math.ceil(etaSeconds / 60))
  const unit = minutes === 1 ? 'minute' : 'minutes'
  const progress = messagesSeen > 0 ? ` (${messagesSeen} messages synced so far)` : ''
  return `Reading your mailbox - usually ready in about ${minutes} ${unit}${progress}.`
}

// Polled from the Shell (App.jsx) so it's visible no matter which page is
// open - a brand-new login lands on /chat, which is exactly where someone
// would otherwise just see silent 409s until the first sync finishes. Stops
// polling once there's nothing left to wait on: `full_history_synced` can
// still be catching up in the background even after `status` flips to
// 'ready' (see ingestion/worker.py's backfill_history), so that's the real
// "done" condition, not just status.
export default function SyncBanner() {
  const [status, setStatus] = useState(null)
  const timer = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const s = await api.syncStatus()
        if (cancelled) return
        setStatus(s)
        if (s.status === 'error') return
        if (s.status !== 'ready' || !s.full_history_synced) {
          timer.current = setTimeout(poll, POLL_MS)
        }
      } catch {
        if (!cancelled) timer.current = setTimeout(poll, POLL_MS)
      }
    }

    poll()
    return () => {
      cancelled = true
      clearTimeout(timer.current)
    }
  }, [])

  if (!status) return null

  if (status.status === 'error') {
    return (
      <div className="sync-banner sync-banner-error">
        Couldn't sync your mailbox: {status.error_detail || 'unknown error'}.
      </div>
    )
  }

  if (status.status !== 'ready') {
    return <div className="sync-banner">{etaMessage(status.eta_seconds, status.messages_seen)}</div>
  }

  if (!status.full_history_synced) {
    return (
      <div className="sync-banner sync-banner-subtle">
        Recent mail is ready to ask about - still importing older mail in the background.
      </div>
    )
  }

  return null
}
