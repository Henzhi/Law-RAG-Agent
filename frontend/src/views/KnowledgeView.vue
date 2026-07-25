<template>
  <div class="knowledge-page">
    <header class="page-header">
      <button class="btn-back" @click="$router.push('/')">← 返回</button>
      <h2>知识库管理</h2>
    </header>

    <div class="upload-section">
      <h3>上传文档</h3>
      <p class="hint">支持 PDF、DOCX、TXT，最大 50MB</p>

      <form @submit.prevent="handleUpload" class="upload-form">
        <div class="form-row">
          <label>
            文档类型
            <select v-model="docType">
              <option value="law">法律</option>
              <option value="interpretation">司法解释</option>
              <option value="case">典型案例</option>
              <option value="regulation">地方法规</option>
            </select>
          </label>
          <label>
            来源
            <input v-model="source" placeholder="如：全国人大" />
          </label>
          <label>
            生效日期
            <input v-model="effectiveDate" type="date" />
          </label>
        </div>

        <div class="file-input-row">
          <label class="file-label">
            <input type="file" accept=".pdf,.docx,.txt" @change="onFileChange" />
            <span class="file-btn">选择文件</span>
          </label>
          <span class="file-name">{{ file ? file.name : '未选择文件' }}</span>
        </div>

        <button type="submit" :disabled="!file || uploading" class="btn-upload">
          {{ uploading ? '上传中...' : '上传文档' }}
        </button>

        <p v-if="uploadError" class="error">{{ uploadError }}</p>
        <p v-if="uploadOk" class="success">{{ uploadOk }}</p>
      </form>
    </div>

    <div v-if="taskId" class="task-section">
      <h3>处理任务</h3>
      <div class="task-card">
        <div class="task-row">
          <span class="task-id">任务: {{ taskId.slice(0, 8) }}...</span>
          <span class="task-status" :class="taskStatus">{{ taskStatusText }}</span>
        </div>
        <div class="progress-bar" v-if="taskProgress > 0">
          <div class="progress-fill" :style="{ width: taskProgress + '%' }"></div>
        </div>
        <div class="progress-text" v-if="taskProgress > 0">{{ taskProgress }}%</div>
        <button v-if="taskDone" class="btn-refresh" @click="$router.push('/')">去问答</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { uploadDocument, getIngestionStatus } from '../api'

const docType = ref('law')
const source = ref('')
const effectiveDate = ref('')
const file = ref(null)
const uploading = ref(false)
const uploadError = ref('')
const uploadOk = ref('')
const taskId = ref('')
const taskStatus = ref('')
const taskProgress = ref(0)

const taskDone = computed(() => taskStatus.value === 'done')
const taskStatusText = computed(() => ({
  pending: '等待处理', parsing: '解析中', chunking: '分块中',
  embedding: '向量化中', indexing: '索引中', done: '完成', failed: '失败'
}[taskStatus.value] || taskStatus.value))

const ALLOWED_TYPES = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
const MAX_SIZE = 50 * 1024 * 1024 // 50MB

function onFileChange(e) {
  const f = e.target.files[0] || null
  uploadError.value = ''
  uploadOk.value = ''
  if (!f) { file.value = null; return }
  // 类型校验
  if (!ALLOWED_TYPES.includes(f.type) && !f.name.match(/\.(pdf|docx|txt)$/i)) {
    uploadError.value = '仅支持 PDF、DOCX、TXT 文件'
    file.value = null
    e.target.value = ''
    return
  }
  // 大小校验
  if (f.size > MAX_SIZE) {
    uploadError.value = `文件过大（${(f.size / 1024 / 1024).toFixed(1)}MB），最大 50MB`
    file.value = null
    e.target.value = ''
    return
  }
  file.value = f
}

async function handleUpload() {
  if (!file.value) return
  uploading.value = true
  uploadError.value = ''
  uploadOk.value = ''
  try {
    const res = await uploadDocument(file.value, docType.value, source.value, effectiveDate.value)
    taskId.value = res.task_id
    taskStatus.value = res.status
    uploadOk.value = res.message
    pollStatus()
  } catch (e) {
    uploadError.value = e.message
  } finally {
    uploading.value = false
  }
}

function pollStatus() {
  const timer = setInterval(async () => {
    try {
      const s = await getIngestionStatus(taskId.value)
      taskStatus.value = s.status
      taskProgress.value = s.progress || 0
      if (s.status === 'done' || s.status === 'failed') {
        clearInterval(timer)
        if (s.status === 'failed') uploadError.value = s.error || '处理失败'
      }
    } catch { /* ignore */ }
  }, 2000)
}
</script>

<style scoped>
.knowledge-page { max-width: 720px; margin: 0 auto; padding: 24px; }
.page-header { display: flex; align-items: center; gap: 16px; margin-bottom: 32px; }
.page-header h2 { font-size: 22px; }
.btn-back { background: none; border: 1px solid var(--color-border); border-radius: var(--radius); padding: 6px 14px; cursor: pointer; color: var(--color-text-muted); font-size: 14px; }
.upload-section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 24px; }
.upload-section h3 { margin-bottom: 8px; font-size: 18px; }
.hint { color: var(--color-text-muted); font-size: 13px; margin-bottom: 20px; }
.upload-form { display: flex; flex-direction: column; gap: 16px; }
.form-row { display: flex; gap: 16px; flex-wrap: wrap; }
.form-row label { flex: 1; min-width: 160px; display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: var(--color-text-muted); }
.form-row input, .form-row select { padding: 8px 12px; border: 1px solid var(--color-border); border-radius: var(--radius); font-size: 14px; background: var(--color-bg); }
.file-input-row { display: flex; align-items: center; gap: 12px; }
.file-label { cursor: pointer; }
.file-label input[type="file"] { display: none; }
.file-btn { display: inline-block; padding: 8px 20px; background: var(--color-primary-light); color: var(--color-primary-dark); border-radius: var(--radius); font-size: 14px; font-weight: 500; }
.file-name { font-size: 13px; color: var(--color-text-muted); }
.btn-upload { padding: 10px 24px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius); font-size: 15px; cursor: pointer; }
.btn-upload:disabled { opacity: 0.5; cursor: not-allowed; }
.error { color: var(--color-error); font-size: 13px; }
.success { color: #059669; font-size: 13px; }
.task-section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 24px; }
.task-section h3 { margin-bottom: 12px; font-size: 18px; }
.task-card { display: flex; flex-direction: column; gap: 12px; }
.task-row { display: flex; justify-content: space-between; align-items: center; }
.task-id { font-family: monospace; font-size: 13px; }
.task-status { font-size: 12px; padding: 3px 10px; border-radius: 10px; background: var(--color-primary-light); color: var(--color-primary-dark); }
.task-status.done { background: #D1FAE5; color: #065F46; }
.task-status.failed { background: #FEE2E2; color: #991B1B; }
.progress-bar { height: 8px; background: var(--color-border); border-radius: 4px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--color-primary); border-radius: 4px; transition: width 0.5s ease; }
.progress-text { font-size: 12px; color: var(--color-text-muted); text-align: right; margin-top: 4px; }
.btn-refresh { padding: 6px 16px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius); cursor: pointer; font-size: 14px; }
</style>
