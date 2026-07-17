<template>
  <div class="input-area">
    <div :class="['input-row', { disabled: disabled }]">
      <textarea
        v-model="text"
        :disabled="disabled"
        rows="1"
        :placeholder="disabled ? 'AI 正在回复中，请稍候...' : '输入法律问题，Enter 发送 / Shift+Enter 换行'"
        @keydown="onKeydown"
        @input="autoResize"
        ref="textareaRef"
      ></textarea>
      <button v-if="disabled" class="btn-sending">
        <span class="dot-pulse"></span>
      </button>
      <button v-else class="btn-send" @click="doSend" :disabled="!text.trim()">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({ disabled: Boolean })
const emit = defineEmits(['send'])

const text = ref('')
const textareaRef = ref(null)

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    doSend()
  }
}

function autoResize() {
  const el = textareaRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 150) + 'px'
}

function doSend() {
  const q = text.value.trim()
  if (!q || props.disabled) return
  emit('send', q)
  text.value = ''
  if (textareaRef.value) {
    textareaRef.value.style.height = 'auto'
  }
}
</script>

<style scoped>
.input-area {
  max-width: 900px;
  margin: 0 auto;
  width: 100%;
  padding: 0 20px 16px;
  flex-shrink: 0;
}

.input-row {
  display: flex;
  gap: 8px;
  align-items: flex-end;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 16px;
  padding: 8px 8px 8px 18px;
  transition: border-color 150ms ease, box-shadow 150ms ease, opacity 200ms ease;
}
.input-row:focus-within {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
}
.input-row.disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

textarea {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  font-size: 15px;
  font-family: var(--font-body);
  line-height: 1.5;
  resize: none;
  padding: 6px 0;
  max-height: 150px;
  color: var(--color-text);
  cursor: text;
}
textarea:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}
textarea::placeholder { color: var(--color-text-muted); }

.btn-send {
  flex-shrink: 0;
  width: 38px;
  height: 38px;
  border-radius: 50%;
  border: none;
  background: var(--color-primary);
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 150ms ease;
}
.btn-send:hover:not(:disabled) { background: var(--color-primary-dark); transform: scale(1.05); }
.btn-send:disabled { background: #D1D5DB; cursor: not-allowed; }

.btn-sending {
  flex-shrink: 0;
  width: 38px;
  height: 38px;
  border-radius: 50%;
  border: 2px solid var(--color-primary);
  background: var(--color-primary-light);
  cursor: default;
  display: flex;
  align-items: center;
  justify-content: center;
}

.dot-pulse {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--color-primary);
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 0.3; transform: scale(0.8); }
  50% { opacity: 1; transform: scale(1.3); }
}
</style>
