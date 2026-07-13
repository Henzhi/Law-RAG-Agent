const BASE = '/api'

function authHeaders() {
  const t = localStorage.getItem('lawrag_token')
  return t ? { Authorization: `Bearer ${t}`, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' }
}

function handleError(r) {
  if (!r.ok) throw new Error(r.status === 401 ? '认证失败，请重新登录' : '请求失败')
  return r.json()
}

// Auth
export const login = (username, password) =>
  fetch(`${BASE}/auth/login`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) }).then(handleError)

export const register = (username, password) =>
  fetch(`${BASE}/auth/register`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) }).then(handleError)

export const getMe = () =>
  fetch(`${BASE}/auth/me`, { headers: authHeaders() }).then(handleError)

// Conversations
export const listConversations = () =>
  fetch(`${BASE}/conversations`, { headers: authHeaders() }).then(handleError)

export const loadHistory = (sessionId) =>
  fetch(`${BASE}/conversations/${sessionId}`, { headers: authHeaders() }).then(handleError)

export const saveSession = (sessionId, messages) =>
  fetch(`${BASE}/conversations/${sessionId}`, { method: 'POST', headers: authHeaders(), body: JSON.stringify({ messages }) })

// Chat Stream
export async function* streamChat(query, history, sessionId, topK = 5) {
  const resp = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ query, history, session_id: sessionId, top_k: topK }),
  })
  if (!resp.ok) throw new Error(`请求失败: ${resp.status}`)

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const data = line.slice(6)
      if (data === '[DONE]') return
      try {
        const msg = JSON.parse(data)
        yield msg
      } catch { /* skip malformed */ }
    }
  }
}
