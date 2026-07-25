<template>
  <div class="history-page">
    <header class="page-header">
      <button class="btn-back" @click="$router.push('/')">← 返回</button>
      <h2>对话历史</h2>
    </header>

    <div v-if="loading" class="loading">加载中...</div>

    <div v-else-if="!sessions.length" class="empty">暂无历史对话</div>

    <div v-else class="session-list">
      <div v-for="s in sessions" :key="s.session_id" class="session-card">
        <div class="session-info">
          <div class="session-id">{{ s.session_id.slice(0, 12) }}...</div>
          <div class="session-meta">
            {{ s.message_count || s.messages?.length || 0 }} 条消息 ·
            {{ formatTime(s.updated_at || s.created_at) }}
          </div>
        </div>
        <div class="session-actions">
          <button class="btn-open" @click="openSession(s.session_id)">继续对话</button>
          <button class="btn-delete" @click="doDelete(s.session_id)">删除</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { listConversations, deleteConversation } from '../api'

const router = useRouter()
const sessions = ref([])
const loading = ref(true)

onMounted(async () => {
  try {
    sessions.value = await listConversations()
  } catch { /* ignore */ }
  finally { loading.value = false }
})

function formatTime(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`
}

function openSession(sid) {
  router.push({ path: '/', query: { session: sid } })
}

async function doDelete(sid) {
  if (!confirm('确定删除此对话？')) return
  try {
    await deleteConversation(sid)
    sessions.value = sessions.value.filter(s => s.session_id !== sid)
  } catch { /* ignore */ }
}
</script>

<style scoped>
.history-page { max-width: 720px; margin: 0 auto; padding: 24px; }
.page-header { display: flex; align-items: center; gap: 16px; margin-bottom: 32px; }
.page-header h2 { font-size: 22px; }
.btn-back { background: none; border: 1px solid var(--color-border); border-radius: var(--radius); padding: 6px 14px; cursor: pointer; color: var(--color-text-muted); font-size: 14px; }
.loading, .empty { text-align: center; color: var(--color-text-muted); padding: 60px 0; font-size: 15px; }
.session-list { display: flex; flex-direction: column; gap: 12px; }
.session-card { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 16px 20px; display: flex; justify-content: space-between; align-items: center; }
.session-info { display: flex; flex-direction: column; gap: 4px; }
.session-id { font-family: monospace; font-size: 14px; color: var(--color-text); }
.session-meta { font-size: 12px; color: var(--color-text-muted); }
.session-actions { display: flex; gap: 8px; }
.btn-open { padding: 6px 16px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius); cursor: pointer; font-size: 13px; }
.btn-delete { padding: 6px 16px; background: none; border: 1px solid var(--color-border); border-radius: var(--radius); cursor: pointer; color: var(--color-error); font-size: 13px; }
</style>
