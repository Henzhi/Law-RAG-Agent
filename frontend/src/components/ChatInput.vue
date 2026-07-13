<template>
  <div class="input-area">
    <button class="btn-clear" @click="$emit('clear')" title="清空对话">清空</button>
    <input class="topk-input" type="number" v-model="topK" min="1" max="20" title="检索条文数" />
    <textarea
      v-model="text"
      rows="1"
      placeholder="输入法律问题，按 Enter 发送，Shift+Enter 换行"
      @keydown="onKeydown"
      ref="textareaRef"
    ></textarea>
    <button class="btn-send" @click="doSend" :disabled="disabled || !text.trim()">发送</button>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({ disabled: Boolean })
const emit = defineEmits(['send', 'clear'])

const text = ref('')
const topK = ref(5)
const textareaRef = ref(null)

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    doSend()
  }
}

function doSend() {
  const q = text.value.trim()
  if (!q || props.disabled) return
  emit('send', q, topK.value)
  text.value = ''
  textareaRef.value?.focus()
}
</script>

<style scoped>
.input-area {
  padding: 16px 20px;
  background: var(--color-surface);
  border-top: 1px solid var(--color-border);
  display: flex;
  gap: 10px;
  max-width: 900px;
  margin: 0 auto;
  width: 100%;
  align-items: flex-end;
  flex-shrink: 0;
}
.btn-clear {
  background: none;
  border: 1px solid var(--color-border);
  color: var(--color-text-muted);
  padding: 10px 16px;
  border-radius: var(--radius);
  cursor: pointer;
  font-size: 14px;
  font-family: var(--font-body);
  white-space: nowrap;
  transition: all var(--transition);
}
.btn-clear:hover { background: #FEF2F2; color: var(--color-error); border-color: var(--color-error); }
.topk-input {
  width: 60px;
  padding: 10px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  font-size: 14px;
  text-align: center;
  outline: none;
  font-family: var(--font-body);
}
.topk-input:focus { border-color: var(--color-primary); }
textarea {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  font-size: 15px;
  resize: none;
  outline: none;
  line-height: 1.5;
  min-height: 44px;
  max-height: 120px;
  font-family: var(--font-body);
  transition: border-color var(--transition);
}
textarea:focus { border-color: var(--color-primary); box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.12); }
.btn-send {
  background: var(--color-primary);
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: var(--radius);
  cursor: pointer;
  font-size: 15px;
  font-family: var(--font-body);
  white-space: nowrap;
  transition: background var(--transition);
}
.btn-send:hover:not(:disabled) { background: var(--color-primary-dark); }
.btn-send:disabled { background: #D1D5DB; cursor: not-allowed; }
</style>
