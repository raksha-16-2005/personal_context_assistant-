import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import EmailModal from '../components/EmailModal'

const LIST_FILTERS = ['pending', 'confirmed', 'dismissed', 'all']
const WEEKDAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

// The user's own local day, not the browser's or the server's - "today" on
// the calendar has to agree with what chat and the digest already call
// today (see Settings.jsx's timezone selector). 'en-CA' is just a cheap way
// to get Intl to hand back YYYY-MM-DD directly.
function todayKeyFor(timeZone) {
  return new Intl.DateTimeFormat('en-CA', { timeZone }).format(new Date())
}

function dateKey(year, month, day) {
  return `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
}

function daysInMonth(year, month) {
  return new Date(year, month + 1, 0).getDate()
}

function firstWeekday(year, month) {
  return new Date(year, month, 1).getDay()
}

function monthLabel(year, month) {
  return new Date(year, month, 1).toLocaleDateString(undefined, { month: 'long', year: 'numeric' })
}

// `owner`/`counterparty` can be a semicolon-separated recipient list pulled
// straight from a message header - sometimes dozens of addresses long. The
// first name plus a count reads better than either the raw string (it can
// run off the card with no natural wrap point) or a mid-string ellipsis
// (which cuts a real address in half).
function truncateNames(value, max = 2) {
  if (!value) return value
  const parts = value.split(';').map((p) => p.trim()).filter(Boolean)
  if (parts.length <= max) return parts.join(', ')
  return `${parts.slice(0, max).join(', ')} +${parts.length - max} more`
}

function formatLongDate(key) {
  const [y, m, d] = key.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString(
    undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })
}

export default function Commitments() {
  const [suggestions, setSuggestions] = useState([])
  const [error, setError] = useState('')
  const [busyId, setBusyId] = useState(null)
  const [view, setView] = useState('calendar')
  const [listFilter, setListFilter] = useState('pending')
  const [tz, setTz] = useState('UTC')
  const [cursor, setCursor] = useState(null)
  const [selectedDate, setSelectedDate] = useState(null)
  const [openMessageId, setOpenMessageId] = useState(null)

  const load = () => api.suggestions('all').then(setSuggestions).catch(() => {})

  useEffect(() => {
    load()
  }, [])

  useEffect(() => {
    api.me().then((me) => me.timezone || 'UTC').catch(() => 'UTC').then((zone) => {
      setTz(zone)
      const key = todayKeyFor(zone)
      const [y, m] = key.split('-').map(Number)
      setCursor({ year: y, month: m - 1 })
      setSelectedDate(key)
    })
  }, [])

  const byDate = useMemo(() => {
    const out = {}
    for (const s of suggestions) {
      if (!s.due_at) continue
      if (!out[s.due_at]) out[s.due_at] = []
      out[s.due_at].push(s)
    }
    return out
  }, [suggestions])

  async function act(id, action) {
    setBusyId(id)
    setError('')
    try {
      if (action === 'confirm') {
        await api.confirmSuggestion(id)
      } else {
        await api.dismissSuggestion(id)
      }
      await load()
    } catch (err) {
      setError(err.message || 'that action failed')
    } finally {
      setBusyId(null)
    }
  }

  function changeMonth(delta) {
    setCursor((c) => {
      let month = c.month + delta
      let year = c.year
      if (month < 0) { month = 11; year -= 1 }
      if (month > 11) { month = 0; year += 1 }
      return { year, month }
    })
  }

  function goToday() {
    const key = todayKeyFor(tz)
    const [y, m] = key.split('-').map(Number)
    setCursor({ year: y, month: m - 1 })
    setSelectedDate(key)
  }

  function renderItemActions(s) {
    return s.status === 'pending' ? (
      <div className="suggestion-actions">
        <button disabled={busyId === s.id} onClick={() => act(s.id, 'confirm')}>
          Keep &mdash; add to calendar
        </button>
        <button disabled={busyId === s.id} onClick={() => act(s.id, 'dismiss')}>
          Remove
        </button>
      </div>
    ) : (
      <span className={`status-badge ${s.status}`}>{s.status}</span>
    )
  }

  function renderItem(s) {
    return (
      <li key={s.id} className="suggestion-card">
        <div className="suggestion-main">
          <strong>{s.text}</strong>
        </div>
        <div className="suggestion-meta">
          {s.owner && <span>{truncateNames(s.owner)}</span>}
          {s.counterparty && <span>&rarr; {truncateNames(s.counterparty)}</span>}
          <span className="kind">{s.kind}</span>
        </div>
        {s.message_id && (
          <button
            type="button"
            className="link-button"
            onClick={() => setOpenMessageId(s.message_id)}
          >
            View full email
          </button>
        )}
        {renderItemActions(s)}
      </li>
    )
  }

  const todayKey = todayKeyFor(tz)

  return (
    <div className="page">
      <div className="page-header">
        <h1>Commitments</h1>
        <div className="filter-tabs">
          <button className={view === 'calendar' ? 'active' : ''} onClick={() => setView('calendar')}>
            Calendar
          </button>
          <button className={view === 'list' ? 'active' : ''} onClick={() => setView('list')}>
            List
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      {view === 'calendar' ? (
        cursor && (
          <CalendarView
            cursor={cursor}
            todayKey={todayKey}
            selectedDate={selectedDate}
            byDate={byDate}
            onChangeMonth={changeMonth}
            onGoToday={goToday}
            onSelectDate={setSelectedDate}
            renderItem={renderItem}
          />
        )
      ) : (
        <>
          <div className="filter-tabs" style={{ marginBottom: 12 }}>
            {LIST_FILTERS.map((f) => (
              <button
                key={f}
                className={f === listFilter ? 'active' : ''}
                onClick={() => setListFilter(f)}
              >
                {f}
              </button>
            ))}
          </div>
          {(() => {
            const rows = listFilter === 'all'
              ? suggestions
              : suggestions.filter((s) => s.status === listFilter)
            return rows.length === 0 ? (
              <p className="empty-state">Nothing here yet.</p>
            ) : (
              <ul className="suggestion-list">{rows.map(renderItem)}</ul>
            )
          })()}
        </>
      )}
      {openMessageId && (
        <EmailModal messageId={openMessageId} onClose={() => setOpenMessageId(null)} />
      )}
    </div>
  )
}

function CalendarView({ cursor, todayKey, selectedDate, byDate, onChangeMonth, onGoToday,
                        onSelectDate, renderItem }) {
  const { year, month } = cursor
  const numDays = daysInMonth(year, month)
  const startWeekday = firstWeekday(year, month)

  const cells = []
  for (let i = 0; i < startWeekday; i++) cells.push(null)
  for (let day = 1; day <= numDays; day++) cells.push(day)

  const selectedItems = selectedDate ? (byDate[selectedDate] || []) : []

  return (
    <div className="calendar-layout">
      <div className="calendar-panel">
        <div className="calendar-header">
          <button onClick={() => onChangeMonth(-1)} aria-label="Previous month">&larr;</button>
          <h2>{monthLabel(year, month)}</h2>
          <button onClick={() => onChangeMonth(1)} aria-label="Next month">&rarr;</button>
          <button className="link-button calendar-today-btn" onClick={onGoToday}>Today</button>
        </div>

        <div className="calendar-legend">
          <span><i className="dot pending" /> pending</span>
          <span><i className="dot confirmed" /> confirmed</span>
          <span><i className="dot dismissed" /> dismissed</span>
        </div>

        <div className="calendar-grid">
          {WEEKDAY_LABELS.map((w) => (
            <div key={w} className="calendar-weekday">{w}</div>
          ))}
          {cells.map((day, i) => {
            if (day === null) return <div key={`blank-${i}`} className="calendar-cell empty" />
            const key = dateKey(year, month, day)
            const items = byDate[key] || []
            const classes = ['calendar-cell']
            if (key === todayKey) classes.push('today')
            if (key === selectedDate) classes.push('selected')
            if (items.length) classes.push('has-items')
            return (
              <button key={key} className={classes.join(' ')} onClick={() => onSelectDate(key)}>
                <span className="cell-day">{day}</span>
                {items.length > 0 && (
                  <span className="cell-dots">
                    {items.slice(0, 3).map((it) => (
                      <span key={it.id} className={`dot ${it.status}`} />
                    ))}
                    {items.length > 3 && <span className="cell-more">+{items.length - 3}</span>}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      <div className="calendar-detail">
        <h2>{selectedDate ? formatLongDate(selectedDate) : 'Pick a day'}</h2>
        {selectedDate && selectedItems.length === 0 && (
          <p className="empty-state">Nothing due this day.</p>
        )}
        {selectedItems.length > 0 && (
          <ul className="suggestion-list">{selectedItems.map(renderItem)}</ul>
        )}
      </div>
    </div>
  )
}
