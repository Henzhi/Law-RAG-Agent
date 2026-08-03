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
  fetch(`${BASE}/conversations/${sessionId}`, { method: 'POST', headers: authHeaders(), body: JSON.stringify({ messages }) }).then(r => { if (!r.ok) throw new Error('会话保存失败') })

export const deleteConversation = (sessionId) =>
  fetch(`${BASE}/conversations/${sessionId}`, { method: 'DELETE', headers: authHeaders() }).then(r => { if (!r.ok) throw new Error('删除失败') })

// Knowledge
export const uploadDocument = async (file, docType, source, effectiveDate, status = 'active') => {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('doc_type', docType)
  fd.append('source', source)
  fd.append('effective_date', effectiveDate)
  fd.append('status', status)
  const resp = await fetch(`${BASE}/knowledge/upload`, { method: 'POST', headers: { Authorization: authHeaders().Authorization }, body: fd })
  if (!resp.ok) throw new Error('上传失败')
  return resp.json()
}

export const getIngestionStatus = (taskId) =>
  fetch(`${BASE}/knowledge/status/${taskId}`, { headers: authHeaders() }).then(handleError)

export const listDocuments = (opts = {}) => {
  const params = new URLSearchParams()
  if (opts.docType) params.set('doc_type', opts.docType)
  if (opts.status) params.set('status', opts.status)
  if (opts.q) params.set('q', opts.q)
  if (opts.sort) params.set('sort', opts.sort)
  if (opts.order) params.set('order', opts.order)
  if (opts.limit != null) params.set('limit', opts.limit)
  if (opts.offset != null) params.set('offset', opts.offset)
  const qs = params.toString()
  return fetch(`${BASE}/knowledge/documents${qs ? `?${qs}` : ''}`, { headers: authHeaders() }).then(handleError)
}

export const deleteDocument = (docId) =>
  fetch(`${BASE}/knowledge/documents/${docId}`, { method: 'DELETE', headers: authHeaders() }).then(r => { if (!r.ok) throw new Error('删除失败'); return r.json() })

export const getDocumentChunks = (docId, limit = 50, offset = 0) =>
  fetch(`${BASE}/knowledge/documents/${docId}/chunks?limit=${limit}&offset=${offset}`, { headers: authHeaders() }).then(handleError)

// Chat Stream
export async function* streamChat(query, history, sessionId) {
  const resp = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ query, history, session_id: sessionId }),
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

// Query rewrite（智能改写 / 案情分析模式）
export async function rewriteQuery(query) {
  const resp = await fetch(`${BASE}/rewrite`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ query }),
  })
  if (!resp.ok) throw new Error(`改写请求失败: ${resp.status}`)
  return resp.json()
}

// Crawl（在线更新法律：国家法律法规数据库增量爬取）
export const listCrawlTypes = () =>
  fetch(`${BASE}/crawl/types`, { headers: authHeaders() }).then(handleError)

export const startCrawl = (params) => {
  const body = {
    source: 'npc',
    doc_type: params.doc_type,
    keyword: params.keyword || '',
    limit: params.limit,
    force: !!params.force,
    subdir: params.subdir || '',
    store: params.store || 'both',
    rebuild: !!params.rebuild,
  }
  return fetch(`${BASE}/crawl`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(body),
  }).then(handleError)
}

export const getCrawlStatus = (taskId) =>
  fetch(`${BASE}/crawl/status/${taskId}`, { headers: authHeaders() }).then(handleError)
