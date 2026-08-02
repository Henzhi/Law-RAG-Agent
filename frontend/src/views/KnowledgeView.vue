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
            <input type="file" accept=".pdf,.docx,.txt" multiple @change="onFileChange" />
            <span class="file-btn">选择文件</span>
          </label>
          <span class="file-name">{{ files.length ? `已选 ${files.length} 个文件` : '未选择文件（可多选）' }}</span>
        </div>

        <!-- 待上传文件清单 -->
        <ul v-if="files.length" class="file-list">
          <li v-for="(f, i) in files" :key="f.name + i" class="file-list-item">
            <span class="file-list-name">{{ f.name }}</span>
            <span class="file-list-size">{{ (f.size / 1024).toFixed(0) }}KB</span>
            <button type="button" class="file-list-remove" @click="removeFile(i)" :disabled="uploading">×</button>
          </li>
        </ul>

        <button type="submit" :disabled="!files.length || uploading" class="btn-upload">
          {{ uploading ? '上传中...' : `开始上传（${files.length} 个文件）` }}
        </button>

        <p v-if="uploadError" class="error">{{ uploadError }}</p>
        <p v-if="uploadOk" class="success">{{ uploadOk }}</p>
      </form>
    </div>

    <!-- 批量上传任务进度 -->
    <div v-if="tasks.length" class="task-section">
      <h3>批量处理进度（{{ doneCount }}/{{ tasks.length }}）</h3>
      <div v-for="(t, i) in tasks" :key="t.task_id || i + '-' + t.file_name" class="task-card">
        <div class="task-row">
          <span class="task-id" :title="t.file_name">{{ t.file_name }}</span>
          <span class="task-status" :class="t.status">{{ taskStatusText(t.status) }}</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: t.progress + '%' }"></div>
        </div>
        <div class="progress-text">{{ t.progress }}%</div>
        <div v-if="t.error" class="task-error">{{ t.error }}</div>
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
            <td class="actions-cell">
              <button class="btn-view" @click="viewDocument(doc)" :disabled="viewing === doc.id">
                {{ viewing === doc.id ? '加载中...' : '查看' }}
              </button>
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

    <!-- 查看原文弹窗 -->
    <div v-if="viewTarget" class="modal-overlay" @click.self="viewTarget = null">
      <div class="modal modal-wide">
        <h3>{{ viewTarget.title }}</h3>
        <div class="view-meta">
          <span v-if="viewTarget.doc_type">类型: {{ typeLabel(viewTarget.doc_type) }}</span>
          <span v-if="viewTarget.source">来源: {{ viewTarget.source }}</span>
          <span v-if="viewTarget.effective_date">生效: {{ viewTarget.effective_date }}</span>
          <span>{{ viewChunks.length }} 个条文块</span>
        </div>
        <div class="view-loading" v-if="viewLoading">正在加载原文...</div>
        <div class="view-empty" v-else-if="!viewChunks.length">该文档暂无内容</div>
        <div v-else class="view-body">
          <div v-for="(c, i) in viewChunks" :key="c.id" class="chunk-item">
            <div class="chunk-head">
              <span class="chunk-index">{{ i + 1 }}</span>
              <span class="chunk-type">{{ chunkTypeLabel(c.chunk_type) }}</span>
            </div>
            <pre class="chunk-content">{{ c.content }}</pre>
          </div>
        </div>
        <div class="modal-actions">
          <button class="btn-cancel" @click="viewTarget = null">关闭</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { uploadDocument, getIngestionStatus, listDocuments, deleteDocument, getDocumentChunks } from '../api'

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

// --- 查看原文 ---
const viewTarget = ref(null)
const viewChunks = ref([])
const viewLoading = ref(false)
const viewing = ref(null)

async function viewDocument(doc) {
  viewing.value = doc.id
  viewTarget.value = doc
  viewChunks.value = []
  viewLoading.value = true
  try {
    const res = await getDocumentChunks(doc.id)
    viewChunks.value = res.chunks || []
  } catch (e) {
    alert('加载原文失败: ' + e.message)
    viewTarget.value = null
  } finally {
    viewLoading.value = false
    viewing.value = null
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

// --- 上传（多文件并行） ---
const showUpload = ref(false)
const docType = ref('law')
const source = ref('')
const effectiveDate = ref('')
const files = ref([])          // 待上传文件数组
const uploading = ref(false)
const uploadError = ref('')
const uploadOk = ref('')
const tasks = ref([])          // 上传/处理任务队列，每文件一个
const MAX_CONCURRENCY = 3      // 并行上传并发上限，避免同时打垮后端

const doneCount = computed(() => tasks.value.filter(t => t.status === 'done' || t.status === 'failed').length)
const taskStatusText = (s) => ({
  pending: '等待处理', parsing: '解析中', chunking: '分块中',
  embedding: '向量化中', indexing: '索引中', done: '完成', failed: '失败'
}[s] || s)

const ALLOWED_TYPES = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
const MAX_SIZE = 50 * 1024 * 1024

function onFileChange(e) {
  uploadError.value = ''
  uploadOk.value = ''
  const list = Array.from(e.target.files || [])
  const valid = []
  for (const f of list) {
    if (!ALLOWED_TYPES.includes(f.type) && !f.name.match(/\.(pdf|docx|txt)$/i)) {
      uploadError.value += `跳过不支持的文件: ${f.name}\n`
      continue
    }
    if (f.size > MAX_SIZE) {
      uploadError.value += `跳过超大文件: ${f.name}（>50MB）\n`
      continue
    }
    valid.push(f)
  }
  files.value = valid
  e.target.value = ''
}

function removeFile(i) {
  if (uploading.value) return
  files.value.splice(i, 1)
}

/** 并发控制器：最多 MAX_CONCURRENCY 个任务同时跑 */
async function mapLimit(items, limit, fn) {
  const results = []
  let idx = 0
  async function worker() {
    while (idx < items.length) {
      const cur = idx++
      results[cur] = await fn(items[cur], cur)
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, worker)
  await Promise.all(workers)
  return results
}

/** 并行上传所有文件，每个文件独立任务并轮询状态 */
async function handleUpload() {
  if (!files.value.length) return
  uploading.value = true
  uploadError.value = ''
  uploadOk.value = ''
  tasks.value = files.value.map((f) => ({
    file_name: f.name,
    task_id: '',
    status: 'pending',
    progress: 0,
  }))

  await mapLimit(files.value, MAX_CONCURRENCY, async (f, i) => {
    const t = tasks.value[i]
    try {
      const res = await uploadDocument(f, docType.value, source.value, effectiveDate.value)
      t.task_id = res.task_id
      t.status = res.status || 'pending'
      uploadOk.value = `已提交 ${f.name}`
      pollTask(t)
    } catch (e) {
      t.status = 'failed'
      t.progress = 0
      t.error = e.message
    }
  })

  uploading.value = false
  files.value = [] // 清空待传列表，任务队列保留展示
}

/** 轮询单个任务状态直到完成/失败 */
function pollTask(t) {
  const timer = setInterval(async () => {
    if (!t.task_id) return
    try {
      const s = await getIngestionStatus(t.task_id)
      t.status = s.status
      t.progress = s.progress || 0
      if (s.status === 'done' || s.status === 'failed') {
        clearInterval(timer)
        if (s.status === 'done') loadDocuments() // 全部完成后刷新一次
        else t.error = s.error || '处理失败'
      }
    } catch { /* ignore */ }
  }, 2000)
}

// --- 工具函数 ---
function typeLabel(t) {
  return { law: '法律', interpretation: '司法解释', case: '典型案例', regulation: '地方法规' }[t] || t
}

function chunkTypeLabel(t) {
  return { article: '法条', case: '案例段落', summary: '章摘要', judgment: '判决要点', guideline: '指导要点' }[t] || (t || '正文')
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
.file-list { list-style: none; margin: 12px 0 0; padding: 0; border: 1px solid var(--color-border); border-radius: var(--radius); overflow: hidden; }
.file-list-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px; font-size: 13px; border-top: 1px solid var(--color-border); }
.file-list-item:first-child { border-top: none; }
.file-list-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-list-size { color: var(--color-text-muted); flex-shrink: 0; }
.file-list-remove { border: none; background: none; color: var(--color-text-muted); font-size: 16px; cursor: pointer; padding: 0 4px; flex-shrink: 0; }
.file-list-remove:hover { color: var(--color-error); }
.file-list-remove:disabled { cursor: not-allowed; opacity: 0.4; }
.btn-upload { padding: 10px 24px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius); font-size: 15px; cursor: pointer; align-self: flex-start; }
.btn-upload:disabled { opacity: 0.5; cursor: not-allowed; }
.error { color: var(--color-error); font-size: 13px; }
.success { color: #059669; font-size: 13px; }

/* Task progress */
.task-section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 24px; }
.task-section h3 { margin-bottom: 12px; font-size: 16px; }
.task-card { display: flex; flex-direction: column; gap: 10px; padding-bottom: 14px; border-bottom: 1px dashed var(--color-border); margin-bottom: 14px; }
.task-card:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.task-error { color: var(--color-error); font-size: 12px; }
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

.actions-cell { white-space: nowrap; }
.btn-view { padding: 4px 12px; border: 1px solid var(--color-border); border-radius: var(--radius); background: var(--color-primary-light); color: var(--color-primary-dark); cursor: pointer; font-size: 13px; margin-right: 6px; }
.btn-view:hover { background: #EDE9FE; }
.btn-view:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-delete { padding: 4px 12px; border: 1px solid #FECACA; border-radius: var(--radius); background: #FEF2F2; color: #DC2626; cursor: pointer; font-size: 13px; }
.btn-delete:hover { background: #FEE2E2; }
.btn-delete:disabled { opacity: 0.5; cursor: not-allowed; }

/* 查看原文 */
.modal-wide { max-width: 760px; width: 92%; }
.view-meta { display: flex; gap: 16px; flex-wrap: wrap; color: var(--color-text-muted); font-size: 13px; margin-bottom: 12px; }
.view-loading { text-align: center; color: var(--color-text-muted); padding: 32px; }
.view-empty { text-align: center; color: var(--color-text-muted); padding: 32px; }
.view-body { max-height: 60vh; overflow-y: auto; padding-right: 4px; }
.chunk-item { margin-bottom: 12px; border: 1px solid var(--color-border); border-radius: 6px; overflow: hidden; }
.chunk-head { display: flex; align-items: center; gap: 8px; padding: 6px 12px; background: var(--color-primary-light); border-bottom: 1px solid var(--color-border); }
.chunk-index { font-weight: 600; color: var(--color-primary-dark); font-size: 13px; }
.chunk-type { font-size: 12px; color: var(--color-text-muted); }
.chunk-content { margin: 0; padding: 12px; font-size: 14px; line-height: 1.8; color: var(--color-text); white-space: pre-wrap; word-break: break-word; font-family: var(--font-body); }

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
