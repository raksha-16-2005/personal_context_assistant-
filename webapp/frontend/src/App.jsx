import { useEffect, useState } from 'react'
import { BrowserRouter, Navigate, NavLink, Route, Routes } from 'react-router-dom'
import { api } from './api'
import SyncBanner from './components/SyncBanner'
import Chat from './pages/Chat'
import Commitments from './pages/Commitments'
import Digest from './pages/Digest'
import Login from './pages/Login'
import Settings from './pages/Settings'

function useCurrentUser() {
  const [state, setState] = useState({ loading: true, user: null })
  useEffect(() => {
    api
      .me()
      .then((user) => setState({ loading: false, user }))
      .catch(() => setState({ loading: false, user: null }))
  }, [])
  return state
}

function Shell({ user, children }) {
  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">Email RAG</span>
        <nav>
          <NavLink to="/chat">Chat</NavLink>
          <NavLink to="/commitments">Commitments</NavLink>
          <NavLink to="/digest">Digest</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </nav>
        <span className="user-email">{user.email}</span>
      </header>
      <SyncBanner />
      <main>{children}</main>
    </div>
  )
}

export default function App() {
  const { loading, user } = useCurrentUser()

  if (loading) return <div className="centered">Loading...</div>

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={user ? <Navigate to="/chat" replace /> : <Login />} />
        <Route
          path="/*"
          element={
            user ? (
              <Shell user={user}>
                <Routes>
                  <Route path="/" element={<Navigate to="/chat" replace />} />
                  <Route path="/chat" element={<Chat />} />
                  <Route path="/commitments" element={<Commitments />} />
                  <Route path="/digest" element={<Digest />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="*" element={<Navigate to="/chat" replace />} />
                </Routes>
              </Shell>
            ) : (
              <Navigate to="/login" replace />
            )
          }
        />
      </Routes>
    </BrowserRouter>
  )
}
