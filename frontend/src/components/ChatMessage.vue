<template>
  <div :class="['msg', message.role]">
    <!-- 思考过程折叠块（从 message.thinking 持久化数据渲染） -->
    <div v-if="message.role === 'assistant' && message.thinking?.length" class="thinking-box">
      <button class="thinking-toggle" @click="thinkingCollapsed = !thinkingCollapsed">
        <svg :class="{ rotated: !thinkingCollapsed }" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><polyline points="9 18 15 12 9 6"/></svg>
        <span>已思考</span>
      </button>
      <div v-if="!thinkingCollapsed" class="thinking-traces">
        <div v-for="(t, i) in message.thinking" :key="i" class="trace-item">{{ t }}</div>
      </div>
    </div>
    <div class="bubble" v-html="renderedContent"></div>
    <div v-if="sources.length" class="sources">
      <div class="src-title">引用条文</div>
      <ul>
        <li v-for="(s, i) in sources" :key="i">{{ s.citation }}</li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  message: { type: Object, required: true },
  thinking: { type: Boolean, default: false },
  sources: { type: Array, default: () => [] },
})

const thinkingCollapsed = ref(true)

const renderedContent = computed(() => {
  const text = props.message.content || ''
  // Bold markdown **text**
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>')
})
</script>

<style scoped>
.msg {
  margin-bottom: 24px;
  animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.msg.user { text-align: right; }
.msg.user .bubble {
  background: var(--color-primary-dark);
  color: #fff;
  display: inline-block;
  max-width: 80%;
  padding: 10px 16px;
  border-radius: 12px 12px 0 12px;
  text-align: left;
  line-height: 1.6;
}
.msg.assistant .bubble {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  max-width: 85%;
  padding: 14px 18px;
  border-radius: 0 12px 12px 12px;
  line-height: 1.7;
  font-family: var(--font-body);
}
.msg.assistant .bubble :deep(strong) { color: var(--color-primary-dark); }
.msg.assistant .bubble :deep(p) { margin: 8px 0; }
.sources {
  margin-top: 8px;
  padding: 8px 12px;
  background: var(--color-primary-light);
  border-radius: 6px;
  font-size: 13px;
  color: var(--color-text-muted);
  border-left: 3px solid var(--color-primary);
}
.src-title { font-weight: 600; color: var(--color-primary-dark); margin-bottom: 4px; }
.sources li { list-style: none; margin: 2px 0; font-family: var(--font-body); }

/* 思考过程折叠块 */
.thinking-box {
  margin-bottom: 8px;
}
.thinking-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  background: none;
  border: none;
  color: var(--color-text-muted);
  font-size: 13px;
  font-family: var(--font-body);
  cursor: pointer;
  padding: 4px 0;
  transition: color 150ms ease;
}
.thinking-toggle:hover { color: var(--color-primary); }
.thinking-toggle svg {
  transition: transform 150ms ease;
  width: 14px; height: 14px;
}
.thinking-toggle svg.rotated { transform: rotate(90deg); }
.thinking-traces {
  margin-top: 4px;
  padding: 8px 12px;
  background: var(--color-primary-light);
  border-left: 3px solid var(--color-primary);
  border-radius: 0 6px 6px 0;
  font-size: 13px;
  color: var(--color-text-muted);
  line-height: 1.6;
}
.trace-item {
  padding: 2px 0;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
