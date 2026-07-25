<template>
  <div class="knowledge-page">
    <header class="page-header">
      <button class="btn-back" @click="$router.push('/')">← 返回问答</button>
      <h2>知识库管理</h2>
      <div class="header-actions">
        <button class="btn-toggle-upload" @click="showUpload = !showUpload">
          {{ showUpload ? '收起上传' : '+ 上传文档' }}
        </button>
        <button class="btn-refresh" @click="loadDocuments" :disabled="loading">刷新</button>
      </div>
    </header>

    <!-- 上传区域 -->
    <div v-if="showUpload" class="upload-section">
      <h3>上传文档</h3>
      <p class="hint">支持 PDF、DOCX、TXT，最大 50MB。上传后自动解析、分块、向量化。</p>

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
          {{ uploading ? '上传中...' : '开始上传' }}
        </button>

        <p v-if="uploadError" class="error">{{ uploadError }}</p>
        <p v-if="uploadOk" class="success">{{ uploadOk }}</p>
      </form>
    </div>

    <!-- 上传任务进度 -->
    <div v-if="taskId" class="task-section">
      <h3>处理进度</h3>
      <div class="task-card">
        <div class="task-row">
          <span class="task-id">任务: {{ taskId.slice(0, 8) }}...</span>
          <span class="task-status" :class="taskStatus">{{ taskStatusText }}</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: taskProgress + '%' }"></div>
        </div>
        <div class="progress-text">{{ taskProgress }}%</div>
      </div>
    </div>

    <!-- 文档列表 -->
    <div class="documents-section">
      <div class="section-header">
        <h3>文档列表 <span class="count">({{ totalDocuments }})</span></h3>
        <select v-model="filterType" @change="loadDocuments" class="filter-select">
          <option value="">全部类型</option>
          <option value="law">法律</option>
          <option value="interpretation">司法解释</option>
          <option value="case">典型案例</option>
          <option value="regulation">地方法规</option>
        </select>
      </div>

      <div v-if="loading" class="loading">加载中...</div>

      <table v-else-if="documents.length > 0" class="doc-table">
        <thead>
          <tr>
            <th>文档名称</th>
            <th>类型</th>
            <th>来源</th>
            <th>块数</th>
            <th>上传时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="doc in documents" :key="doc.id">
            <td class="title-cell" :title="doc.title">{{ doc.title }}</td>
            <td><span class="type-badge" :class="doc.doc_type">{{ typeLabel(doc.doc_type) }}</span></td>
            <td class="source-cell">{{ doc.source || '-' }}</td>
            <td class="num-cell">{{ doc.chunks }}</td>
            <td class="date-cell">{{ formatDate(doc.created_at) }}</td>
            <td>
              <button class="btn-delete" @click="confirmDelete(doc)" :disabled="deleting === doc.id">
                {{ deleting === doc.id ? '删除中...' : '删除' }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>

      <div v-else class="empty">暂无文档，请上传法律文件开始构建知识库</div>
    </div>

    <!-- 删除确认弹窗 -->
    <div v-if="deleteTarget" class="modal-overlay" @click.self="deleteTarget = null">
      <div class="modal">
        <h3>确认删除</h3>
        <p>确定要删除文档 <strong>{{ deleteTarget.title }}</strong> 吗？</p>
        <p class="warn">此操作将同时删除该文档的 {{ deleteTarget.chunks }} 个向量块，不可恢复。</p>
        <div class="modal-actions">
          <button class="btn-cancel" @click="deleteTarget = null">取消</button>
          <button class="btn-confirm-delete" @click="doDelete" :disabled="deleting">
            {{ deleting ? '删除中...' : '确认删除' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { uploadDocument, getIngestionStatus, listDocuments, deleteDocument } from '../api'

// --- 文档列表 ---
const documents = ref([])
const loading = ref(false)
const filterType = ref('')
const totalDocuments = computed(() => documents.value.length)

async function loadDocuments() {
  loading.value = true
  try {
    const res = await listDocuments(filterType.value || undefined)
    documents.value = res.documents || []
  } catch (e) {
    console.error('加载文档列表失败:', e)
  } finally {
    loading.value = false
  }
}

// --- 删除 ---
const deleteTarget = ref(null)
const deleting = ref(null)

function confirmDelete(doc) {
  deleteTarget.value = doc
}

async function doDelete() {
  if (!deleteTarget.value) return
  const id = deleteTarget.value.id
  deleting.value = id
  try {
    await deleteDocument(id)
    deleteTarget.value = null
    await loadDocuments()
  } catch (e) {
    alert('删除失败: ' + e.message)
  } finally {
    deleting.value = null
  }
}

// --- 上传 ---
const showUpload = ref(false)
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
const MAX_SIZE = 50 * 1024 * 1024

function onFileChange(e) {
  const f = e.target.files[0] || null
  uploadError.value = ''
  uploadOk.value = ''
  if (!f) { file.value = null; return }
  if (!ALLOWED_TYPES.includes(f.type) && !f.name.match(/\.(pdf|docx|txt)$/i)) {
    uploadError.value = '仅支持 PDF、DOCX、TXT 文件'
    file.value = null
    e.target.value = ''
    return
  }
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
    taskProgress.value = 0
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
        if (s.status === 'done') {
          uploadOk.value = '解析完成！已添加到知识库'
          loadDocuments() // 刷新列表
        }
        if (s.status === 'failed') uploadError.value = s.error || '处理失败'
      }
    } catch { /* ignore */ }
  }, 2000)
}

// --- 工具函数 ---
function typeLabel(t) {
  return { law: '法律', interpretation: '司法解释', case: '典型案例', regulation: '地方法规' }[t] || t
}

function formatDate(d) {
  if (!d) return '-'
  try {
    return new Date(d).toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' })
  } catch { return d.slice(0, 10) }
}

onMounted(() => {
  loadDocuments()
})
</script>

<style scoped>
.knowledge-page { max-width: 960px; margin: 0 auto; padding: 24px; }

/* Header */
.page-header { display: flex; align-items: center; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.page-header h2 { font-size: 22px; flex: 1; }
.header-actions { display: flex; gap: 8px; }
.btn-back { background: none; border: 1px solid var(--color-border); border-radius: var(--radius); padding: 6px 14px; cursor: pointer; color: var(--color-text-muted); font-size: 14px; }
.btn-toggle-upload { padding: 8px 18px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius); font-size: 14px; cursor: pointer; }
.btn-refresh { background: none; border: 1px solid var(--color-border); border-radius: var(--radius); padding: 6px 14px; cursor: pointer; color: var(--color-text-muted); font-size: 14px; }
.btn-refresh:disabled { opacity: 0.5; }

/* Upload */
.upload-section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 24px; }
.upload-section h3 { margin-bottom: 8px; font-size: 18px; }
.hint { color: var(--color-text-muted); font-size: 13px; margin-bottom: 20px; }
.upload-form { display: flex; flex-direction: column; gap: 16px; }
.form-row { display: flex; gap: 16px; flex-wrap: wrap; }
.form-row label { flex: 1; min-width: 160px; display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: var(--color-text-muted); }
.form-row input, .form-row select { padding: 8px 12px; border: 1px solid var(--color-border); border-radius: var(--radius); font-size: 14px; background: var(--color-bg); color: var(--color-text); }
.file-input-row { display: flex; align-items: center; gap: 12px; }
.file-label { cursor: pointer; }
.file-label input[type="file"] { display: none; }
.file-btn { display: inline-block; padding: 8px 20px; background: var(--color-primary-light); color: var(--color-primary-dark); border-radius: var(--radius); font-size: 14px; font-weight: 500; }
.file-name { font-size: 13px; color: var(--color-text-muted); }
.btn-upload { padding: 10px 24px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius); font-size: 15px; cursor: pointer; align-self: flex-start; }
.btn-upload:disabled { opacity: 0.5; cursor: not-allowed; }
.error { color: var(--color-error); font-size: 13px; }
.success { color: #059669; font-size: 13px; }

/* Task progress */
.task-section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 24px; }
.task-section h3 { margin-bottom: 12px; font-size: 16px; }
.task-card { display: flex; flex-direction: column; gap: 10px; }
.task-row { display: flex; justify-content: space-between; align-items: center; }
.task-id { font-family: monospace; font-size: 13px; }
.task-status { font-size: 12px; padding: 3px 10px; border-radius: 10px; background: var(--color-primary-light); color: var(--color-primary-dark); }
.task-status.done { background: #D1FAE5; color: #065F46; }
.task-status.failed { background: #FEE2E2; color: #991B1B; }
.progress-bar { height: 8px; background: var(--color-border); border-radius: 4px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--color-primary); border-radius: 4px; transition: width 0.5s ease; }
.progress-text { font-size: 12px; color: var(--color-text-muted); text-align: right; }

/* Documents list */
.documents-section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 24px; }
.section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.section-header h3 { font-size: 18px; }
.count { color: var(--color-text-muted); font-size: 14px; font-weight: normal; }
.filter-select { padding: 6px 12px; border: 1px solid var(--color-border); border-radius: var(--radius); font-size: 13px; background: var(--color-bg); color: var(--color-text); }
.loading { text-align: center; color: var(--color-text-muted); padding: 32px; }
.empty { text-align: center; color: var(--color-text-muted); padding: 48px 16px; font-size: 14px; }

.doc-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.doc-table th { text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--color-border); color: var(--color-text-muted); font-weight: 600; font-size: 13px; white-space: nowrap; }
.doc-table td { padding: 12px; border-bottom: 1px solid var(--color-border); vertical-align: middle; }
.doc-table tr:hover { background: var(--color-primary-light); }
.title-cell { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }
.source-cell { max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--color-text-muted); }
.num-cell { text-align: center; }
.date-cell { color: var(--color-text-muted); white-space: nowrap; font-size: 13px; }

.type-badge { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 500; }
.type-badge.law { background: #EDE9FE; color: #5B21B6; }
.type-badge.interpretation { background: #DBEAFE; color: #1E40AF; }
.type-badge.case { background: #FEF3C7; color: #92400E; }
.type-badge.regulation { background: #D1FAE5; color: #065F46; }

.btn-delete { padding: 4px 12px; border: 1px solid #FECACA; border-radius: var(--radius); background: #FEF2F2; color: #DC2626; cursor: pointer; font-size: 13px; }
.btn-delete:hover { background: #FEE2E2; }
.btn-delete:disabled { opacity: 0.5; cursor: not-allowed; }

/* Modal */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 100; }
.modal { background: var(--color-surface); border-radius: var(--radius-lg); padding: 28px; max-width: 420px; width: 90%; box-shadow: 0 20px 60px rgba(0,0,0,0.15); }
.modal h3 { margin-bottom: 12px; font-size: 18px; }
.modal p { margin-bottom: 8px; color: var(--color-text-muted); font-size: 14px; }
.modal .warn { color: var(--color-error); font-size: 13px; margin-top: 12px; }
.modal-actions { display: flex; gap: 12px; justify-content: flex-end; margin-top: 20px; }
.btn-cancel { padding: 8px 20px; border: 1px solid var(--color-border); border-radius: var(--radius); background: var(--color-bg); cursor: pointer; font-size: 14px; }
.btn-confirm-delete { padding: 8px 20px; border: none; border-radius: var(--radius); background: var(--color-error); color: #fff; cursor: pointer; font-size: 14px; }
.btn-confirm-delete:disabled { opacity: 0.5; }
</style>
