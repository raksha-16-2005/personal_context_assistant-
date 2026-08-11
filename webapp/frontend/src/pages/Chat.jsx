import { useEffect, useState } from 'react'
import { api, ApiError } from '../api'
import EmailModal from '../components/EmailModal'

// The backend's own answer text for a refusal is the literal
// INSUFFICIENT_CONTEXT sentinel - deliberately machine-checkable rather
// than prose (see emailrag.generation.synthesize's docstring), which makes
// it exactly the wrong thing to show a person verbatim in a chat bubble.
// `refused` is the boolean the backend already computed for this - shown in
// its place, not derived by matching the sentinel string here too.
const REFUSAL_MESSAGE =
  "I couldn't find anything in your mailbox that answers that."

function displayText(message) {
  return message.refused ? REFUSAL_MESSAGE : message.content
}

// The model is told to write at most four sentences of flowing prose (see
// synthesize.py's SYSTEM prompt) - correct for a search result, but dense to
// scan in a chat bubble. Splitting into one bullet per line is display-only:
// the stored answer text, its citation markers and the eval harness that
// reads it are all untouched.
//
// Split right after a citation marker's closing "]." rather than after any
// period - a generic sentence-boundary split breaks on "Pvt." or "Ltd." mid-
// name, chopping one claim into two ugly fragments. The system prompt
// requires a citation on every factual claim, so "].. " is where a real
// clause boundary actually is; a rare uncited aside just rides along with
// its neighboring citation's bullet instead of getting its own, which is a
// fine trade for never mangling an abbreviation.
function splitSentences(text) {
  return text.split(/(?<=\]\.)\s+/).map((s) => s.trim()).filter(Boolean)
}

function AnswerBody({ message }) {
  if (message.role !== 'assistant' || message.refused) {
    return <div className="message-content">{displayText(message)}</div>
  }
  const sentences = splitSentences(message.content)
  if (sentences.length < 2) {
    return <div className="message-content">{message.content}</div>
  }
  return (
    <ul className="answer-points">
      {sentences.map((s, i) => (
        <li key={i}>{s}</li>
      ))}
    </ul>
  )
}

export default function Chat() {
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [openMessageId, setOpenMessageId] = useState(null)

  const refreshList = () => api.conversations().then(setConversations).catch(() => {})

  useEffect(() => {
    refreshList()
  }, [])

  useEffect(() => {
    if (!activeId) {
      setMessages([])
      return
    }
    api.conversation(activeId).then(setMessages).catch(() => setMessages([]))
  }, [activeId])

  async function send(e) {
    e.preventDefault()
    const asked = question.trim()
    if (!asked || sending) return
    setSending(true)
    setError('')
    try {
      const resp = await api.chat(asked, activeId)
      setQuestion('')
      setActiveId(resp.conversation_id)
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: asked },
        { role: 'assistant', content: resp.answer, citations: resp.citations,
          refused: resp.refused },
      ])
      refreshList()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'something went wrong')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="chat-layout">
      <aside className="conversation-list">
        <button className="new-chat" onClick={() => setActiveId(null)}>
          + New conversation
        </button>
        {conversations.map((c) => (
          <button
            key={c.id}
            className={`conversation-item ${c.id === activeId ? 'active' : ''}`}
            onClick={() => setActiveId(c.id)}
          >
            {c.title || 'Untitled'}
          </button>
        ))}
      </aside>
      <section className="chat-panel">
        <div className="messages">
          {messages.length === 0 && (
            <p className="empty-state">Ask something about your mailbox.</p>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              <AnswerBody message={m} />
              {!m.refused && m.citations?.length > 0 && (
                <ul className="citations">
                  {m.citations.map((c, ci) => (
                    <li key={ci}>
                      <button
                        type="button"
                        className="citation-link"
                        onClick={() => setOpenMessageId(c.message_id)}
                      >
                        [{c.n}] {c.sender} - {c.subject || '(no subject)'}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
        {error && <p className="error">{error}</p>}
        <form className="composer" onSubmit={send}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your mailbox..."
            disabled={sending}
          />
          <button type="submit" disabled={sending}>
            {sending ? 'Asking...' : 'Ask'}
          </button>
        </form>
      </section>
      {openMessageId && (
        <EmailModal messageId={openMessageId} onClose={() => setOpenMessageId(null)} />
      )}
    </div>
  )
}
