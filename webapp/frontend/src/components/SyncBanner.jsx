import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

const POLL_MS = 4000

function formatDuration(seconds) {
  if (seconds <= 1) return 'a few seconds'
  if (seconds < 60) return `${seconds} seconds`
  const minutes = Math.round(seconds / 60)
  return `${minutes} minute${minutes === 1 ? '' : 's'}`
}

// eta_is_estimate is true only while there's no real progress signal yet
// (the first poll or two of a sync, before ingestion/worker.py's
// on_progress callback has landed) - the backend falls back to a flat
// guess rather than fabricate a number, and this mirrors that honestly
// instead of stating a guess as if it were computed.
function etaMessage(status) {
  const { eta_seconds, eta_is_estimate, progress_current, progress_total } = status
  const time = formatDuration(eta_seconds)
  if (eta_is_estimate) {
    return `Reading your mailbox - usually ready in about ${time}.`
  }
  const progress = progress_total > 0 ? ` (${progress_current} of ${progress_total} messages)` : ''
  return `Reading your mailbox${progress} - about ${time} left.`
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
    return <div className="sync-banner">{etaMessage(status)}</div>
  }

  if (!status.full_history_synced) {
    const progress = status.progress_total > 0
      ? ` (${status.progress_current} of ${status.progress_total} messages)` : ''
    return (
      <div className="sync-banner sync-banner-subtle">
        Recent mail is ready to ask about - still importing older mail in the
        background{progress}.
      </div>
    )
  }

  return null
}
