<template>
  <aside :class="['sidebar', { collapsed: !open }]">
    <div class="sidebar-inner">
      <!-- New Chat -->
      <button class="btn-new-chat" @click="$emit('new-chat')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="17" height="17"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        <span v-if="open">新建对话</span>
      </button>

      <!-- Session list -->
      <div v-if="open" class="session-list">
        <div v-if="sessions.length === 0" class="empty">暂无对话记录</div>
        <button
          v-for="s in sessions"
          :key="s.session_id"
          :class="['session-item', { active: s.session_id === activeId }]"
          @click="$emit('select', s.session_id)"
        >
          <svg class="session-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="15" height="15"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          <span class="session-label">{{ s.first_msg || '新对话' }}</span>
        </button>
      </div>
    </div>

    <!-- Toggle -->
    <button class="toggle-btn" @click="$emit('toggle')" :title="open ? '收起侧栏' : '展开侧栏'">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
        <polyline v-if="open" points="15 18 9 12 15 6" />
        <polyline v-else points="9 18 15 12 9 6" />
      </svg>
    </button>
  </aside>
</template>

<script setup>
defineProps({
  sessions: { type: Array, default: () => [] },
  activeId: { type: String, default: '' },
  open: { type: Boolean, default: true },
})
defineEmits(['new-chat', 'select', 'toggle'])
</script>

<style scoped>
.sidebar {
  position: relative;
  background: #1E1B4B;
  color: #C4B5FD;
  display: flex;
  flex-direction: column;
  transition: width 200ms ease;
  width: 260px;
  flex-shrink: 0;
  overflow: hidden;
}
.sidebar.collapsed { width: 52px; }
.sidebar-inner {
  flex: 1;
  overflow-y: auto;
  padding: 12px 8px;
}
.btn-new-chat {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px;
  color: #E2E8F0;
  cursor: pointer;
  font-size: 14px;
  font-family: inherit;
  transition: all 150ms ease;
  white-space: nowrap;
}
.btn-new-chat:hover { background: rgba(255,255,255,0.14); }
.collapsed .btn-new-chat { justify-content: center; padding: 10px; }

.session-list { margin-top: 12px; }
.empty { padding: 20px 8px; font-size: 13px; color: #6B7280; text-align: center; }

.session-item {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  background: none;
  border: none;
  border-radius: 6px;
  color: #9CA3AF;
  cursor: pointer;
  font-size: 13px;
  font-family: inherit;
  text-align: left;
  transition: all 150ms ease;
  margin-bottom: 2px;
}
.session-item:hover { background: rgba(255,255,255,0.06); color: #E2E8F0; }
.session-item.active { background: rgba(124,58,237,0.2); color: #C4B5FD; }
.session-icon { flex-shrink: 0; color: #6B7280; }
.session-item.active .session-icon { color: #A78BFA; }
.session-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.toggle-btn {
  position: absolute;
  bottom: 16px;
  right: 8px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 6px;
  color: #6B7280;
  cursor: pointer;
  padding: 6px;
  transition: all 150ms ease;
}
.toggle-btn:hover { color: #C4B5FD; background: rgba(255,255,255,0.1); }
.collapsed .toggle-btn { right: 12px; bottom: 16px; }
</style>
