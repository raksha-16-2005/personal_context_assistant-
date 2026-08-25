import { API_BASE } from '../api'

export default function Login() {
  return (
    <div className="login-screen">
      <div className="login-card">
        <h1>Email RAG</h1>
        <p>Ask questions over your own mailbox - retrieval and generation run
          against your own index, answered with your own pasted Gemini key.</p>
        <a className="google-button" href={`${API_BASE}/auth/google/login`}>Sign in with Google</a>
      </div>
    </div>
  )
}
