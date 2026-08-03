<template>
  <div class="knowledge-page">
    <header class="page-header">
      <button class="btn-back" @click="$router.push('/')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
        返回问答
      </button>
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
              <option value="regulation">行政法规</option>
              <option value="judicial_interpretation">司法解释</option>
              <option value="local_regulation">地方性法规</option>
              <option value="constitution">宪法</option>
              <option value="supervision">监察法规</option>
              <option value="case">典型案例</option>
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
          <label>
            效力状态
            <select v-model="docStatus">
              <option value="active">现行有效</option>
              <option value="repealed">已废止</option>
              <option value="revised">已修改</option>
              <option value="pending">尚未生效</option>
            </select>
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

    <!-- 在线更新法律（爬虫） -->
    <div class="crawl-section">
      <div class="section-header">
        <h3>在线更新法律</h3>
        <span class="crawl-hint">数据源：国家法律法规数据库，增量去重，副本自动保存到 LawData/</span>
      </div>
      <div class="crawl-form">
        <div class="form-row">
          <label>
            文档类型
            <select v-model="crawlForm.doc_type">
              <option v-for="(label, key) in crawlTypes" :key="key" :value="key">{{ label }}</option>
            </select>
          </label>
          <label>
            输出目标
            <select v-model="crawlForm.store">
              <option value="both">pgvector + LawData 副本（推荐）</option>
              <option value="pg">仅 pgvector</option>
              <option value="txt">仅 LawData 文本副本</option>
            </select>
          </label>
          <label>
            最多条数
            <input v-model.number="crawlForm.limit" type="number" min="0" max="1000" />
            <span class="field-hint">0 = 不限</span>
          </label>
          <label>
            标题关键词
            <input v-model="crawlForm.keyword" placeholder="空 = 该类型全部，如：数据安全法" />
          </label>
        </div>
        <div class="crawl-options">
          <label class="checkbox-label">
            <input type="checkbox" v-model="crawlForm.force" /> 强制重爬（覆盖已存在文档）
          </label>
          <label class="checkbox-label">
            <input type="checkbox" v-model="crawlForm.rebuild" /> 爬完后重建向量索引
          </label>
        </div>
        <div class="crawl-actions">
          <button class="btn-crawl" @click="handleCrawl" :disabled="crawlRunning || !Object.keys(crawlTypes).length">
            {{ crawlRunning ? '爬取中...' : '开始增量更新' }}
          </button>
          <span v-if="crawlMsg" class="success">{{ crawlMsg }}</span>
          <span v-if="crawlErr" class="error">{{ crawlErr }}</span>
        </div>
      </div>

      <!-- 爬取任务进度 -->
      <div v-if="crawlTask" class="crawl-task">
        <div class="task-row">
          <span class="task-id">任务 {{ crawlTask.task_id }}</span>
          <span class="task-status" :class="crawlTask.status">{{ crawlStatusText(crawlTask.status) }}</span>
        </div>
        <div v-if="crawlTask.progress" class="crawl-progress">
          <span>命中 {{ crawlTask.progress.total }}</span>
          <span class="crawl-add">新增 {{ crawlTask.progress.added }}</span>
          <span class="crawl-upd">更新 {{ crawlTask.progress.updated }}</span>
          <span>跳过 {{ crawlTask.progress.skipped }}</span>
          <span class="crawl-fail">失败 {{ crawlTask.progress.failed }}</span>
        </div>
        <div v-if="crawlTask.errors && crawlTask.errors.length" class="task-errors">
          <div v-for="(e, i) in crawlTask.errors.slice(0, 10)" :key="i" class="task-error">- {{ e }}</div>
          <div v-if="crawlTask.errors.length > 10" class="task-error">… 共 {{ crawlTask.errors.length }} 条错误</div>
        </div>
      </div>
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
          <option value="regulation">行政法规</option>
          <option value="judicial_interpretation">司法解释</option>
          <option value="local_regulation">地方性法规</option>
          <option value="constitution">宪法</option>
          <option value="supervision">监察法规</option>
          <option value="case">典型案例</option>
        </select>
        <select v-model="filterStatus" @change="loadDocuments" class="filter-select">
          <option value="">全部效力</option>
          <option value="active">现行有效</option>
          <option value="repealed">已废止</option>
          <option value="revised">已修改</option>
          <option value="pending">尚未生效</option>
        </select>
      </div>

      <div v-if="loading" class="loading">加载中...</div>

      <table v-else-if="documents.length > 0" class="doc-table">
        <thead>
          <tr>
            <th>文档名称</th>
            <th>类型</th>
            <th>效力</th>
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
            <td><span class="status-badge" :class="doc.status || 'active'">{{ statusLabel(doc.status) }}</span></td>
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

    <!-- 查看原文弹窗（滚动懒加载） -->
    <div v-if="viewTarget" class="modal-overlay" @click.self="closeView">
      <div class="modal modal-wide">
        <h3>{{ viewTarget.title }}</h3>
        <div class="view-meta">
          <span v-if="viewTarget.doc_type">类型: {{ typeLabel(viewTarget.doc_type) }}</span>
          <span v-if="viewTarget.source">来源: {{ viewTarget.source }}</span>
          <span v-if="viewTarget.effective_date">生效: {{ viewTarget.effective_date }}</span>
          <span>已加载 {{ viewChunks.length }} / {{ viewTotal }} 个条文块</span>
        </div>
        <div class="view-loading" v-if="viewLoading">正在加载原文...</div>
        <div class="view-empty" v-else-if="!viewChunks.length">该文档暂无内容</div>
        <div v-else ref="viewBodyEl" class="view-body" @scroll="onViewScroll">
          <div v-for="(c, i) in viewChunks" :key="c.id" class="chunk-item">
            <div class="chunk-head">
              <span class="chunk-index">{{ i + 1 }}</span>
              <span class="chunk-type">{{ chunkTypeLabel(c.chunk_type) }}</span>
            </div>
            <pre class="chunk-content">{{ c.content }}</pre>
          </div>
          <div class="view-more">
            <span v-if="viewLoadingMore" class="view-more-loading">加载中...</span>
            <span v-else-if="viewChunks.length < viewTotal" class="view-more-hint">向下滚动加载更多</span>
            <span v-else class="view-more-done">已全部加载（{{ viewTotal }} 条）</span>
          </div>
        </div>
        <div class="modal-actions">
          <button class="btn-cancel" @click="closeView">关闭</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { uploadDocument, getIngestionStatus, listDocuments, deleteDocument, getDocumentChunks, listCrawlTypes, startCrawl, getCrawlStatus } from '../api'

// --- 文档列表 ---
const documents = ref([])
const loading = ref(false)
const filterType = ref('')
const filterStatus = ref('')
const totalDocuments = computed(() => documents.value.length)

async function loadDocuments() {
  loading.value = true
  try {
    const res = await listDocuments(
      filterType.value || undefined,
      filterStatus.value || undefined,
    )
    documents.value = res.documents || []
  } catch (e) {
    console.error('加载文档列表失败:', e)
  } finally {
    loading.value = false
  }
}

// --- 查看原文（滚动懒加载分页） ---
const CHUNK_PAGE_SIZE = 50  // 每页条文块数量
const viewTarget = ref(null)
const viewChunks = ref([])
const viewLoading = ref(false)
const viewLoadingMore = ref(false)
const viewTotal = ref(0)
const viewing = ref(null)
const viewBodyEl = ref(null)
let viewOffset = 0

async function viewDocument(doc) {
  viewing.value = doc.id
  viewTarget.value = doc
  viewChunks.value = []
  viewTotal.value = doc.chunks || 0
  viewOffset = 0
  viewLoading.value = true
  try {
    const res = await getDocumentChunks(doc.id, CHUNK_PAGE_SIZE, 0)
    viewChunks.value = res.chunks || []
    if (res.total != null) viewTotal.value = res.total
  } catch (e) {
    alert('加载原文失败: ' + e.message)
    viewTarget.value = null
  } finally {
    viewLoading.value = false
    viewing.value = null
  }
}

// 滚动触底加载下一页（一次性请求，防止重复触发）
async function loadMoreChunks() {
  if (viewLoadingMore.value || viewLoading.value) return
  if (viewChunks.value.length >= viewTotal.value) return
  viewLoadingMore.value = true
  const nextOffset = viewChunks.value.length
  try {
    const res = await getDocumentChunks(viewTarget.value.id, CHUNK_PAGE_SIZE, nextOffset)
    const more = res.chunks || []
    if (more.length) {
      viewChunks.value = viewChunks.value.concat(more)
      viewOffset = nextOffset + more.length
    }
  } catch (e) {
    console.error('加载更多条文失败:', e)
  } finally {
    viewLoadingMore.value = false
  }
}

function onViewScroll() {
  const el = viewBodyEl.value
  if (!el) return
  // 距离底部不足 120px 时触发加载
  if (el.scrollTop + el.clientHeight >= el.scrollHeight - 120) {
    loadMoreChunks()
  }
}

function closeView() {
  viewTarget.value = null
  viewChunks.value = []
  viewTotal.value = 0
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
const docStatus = ref('active')
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
      const res = await uploadDocument(f, docType.value, source.value, effectiveDate.value, docStatus.value)
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

// --- 在线更新法律（爬虫：国家法律法规数据库，增量） ---
const crawlTypes = ref({})
const crawlForm = ref({ doc_type: 'law', store: 'both', limit: 50, keyword: '', force: false, rebuild: false })
const crawlTask = ref(null)
const crawlRunning = ref(false)
const crawlMsg = ref('')
const crawlErr = ref('')
let crawlTimer = null

async function loadCrawlTypes() {
  try {
    const res = await listCrawlTypes()
    crawlTypes.value = res.types || {}
  } catch (e) {
    console.error('加载爬取类型失败:', e)
  }
}

async function handleCrawl() {
  crawlRunning.value = true
  crawlMsg.value = ''
  crawlErr.value = ''
  try {
    const res = await startCrawl(crawlForm.value)
    crawlTask.value = {
      task_id: res.task_id,
      status: res.status,
      progress: { total: 0, added: 0, updated: 0, skipped: 0, failed: 0 },
      errors: [],
    }
    crawlMsg.value = '任务已提交，正在增量更新...'
    pollCrawl()
  } catch (e) {
    crawlErr.value = e.message
    crawlRunning.value = false
  }
}

function pollCrawl() {
  if (crawlTimer) clearInterval(crawlTimer)
  crawlTimer = setInterval(async () => {
    if (!crawlTask.value || !crawlTask.value.task_id) return
    try {
      const s = await getCrawlStatus(crawlTask.value.task_id)
      crawlTask.value = { ...crawlTask.value, ...s }
      if (s.status === 'done' || s.status === 'error' || s.status === 'failed') {
        clearInterval(crawlTimer)
        crawlTimer = null
        crawlRunning.value = false
        crawlMsg.value = s.status === 'done' ? '更新完成，文档列表已刷新' : '更新失败'
        if (s.status === 'done') loadDocuments()
      }
    } catch { /* 忽略瞬时错误，下轮重试 */ }
  }, 3000)
}

function crawlStatusText(s) {
  return { pending: '等待中', running: '爬取中', done: '已完成', error: '失败' }[s] || s
}

// --- 工具函数 ---
function typeLabel(t) {
  // flk 顶级分类规范值 + 历史旧值兼容（judicial/interpretation/local）
  return {
    law: '法律', regulation: '行政法规',
    judicial_interpretation: '司法解释', local_regulation: '地方性法规',
    constitution: '宪法', supervision: '监察法规', case: '典型案例',
    // 历史数据旧值
    interpretation: '司法解释', judicial: '司法解释', local: '地方性法规',
  }[t] || t
}

// 效力状态中文名（历史数据无 status 时默认显示现行有效）
function statusLabel(s) {
  return { active: '现行有效', repealed: '已废止', revised: '已修改', pending: '尚未生效' }[s || 'active'] || (s || '现行有效')
}

function chunkTypeLabel(t) {
  return {
    article: '法条', case: '案例段落', summary: '章摘要',
    judgment: '判决要点', guideline: '指导要点',
    constitution: '宪法条文', law: '法律条文', regulation: '行政法规条文',
    supervision: '监察法规条文', local_regulation: '地方法规条文',
    judicial_interpretation: '司法解释段落', interpretation: '司法解释段落',
  }[t] || (t || '正文')
}

function formatDate(d) {
  if (!d) return '-'
  try {
    return new Date(d).toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' })
  } catch { return d.slice(0, 10) }
}

onMounted(() => {
  loadDocuments()
  loadCrawlTypes()
})

onBeforeUnmount(() => {
  if (crawlTimer) clearInterval(crawlTimer)
})
</script>

<style scoped>
.knowledge-page { max-width: 960px; margin: 0 auto; padding: 24px; }

/* Header */
.page-header { display: flex; align-items: center; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.page-header h2 { font-size: 22px; flex: 1; }
.header-actions { display: flex; gap: 8px; }
.btn-back { background: none; border: 1px solid var(--color-border); border-radius: var(--radius); padding: 7px 14px; cursor: pointer; color: var(--color-text-secondary); font-size: 14px; display: inline-flex; align-items: center; gap: 6px; transition: all var(--transition); }
.btn-back:hover { background: var(--color-sidebar-hover); color: var(--color-primary); }
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
.form-row input, .form-row select { padding: 8px 12px; border: 1px solid var(--color-border); border-radius: var(--radius); font-size: 14px; background: var(--color-surface-soft); color: var(--color-text); }
.file-input-row { display: flex; align-items: center; gap: 12px; }
.file-label { cursor: pointer; }
.file-label input[type="file"] { display: none; }
.file-btn { display: inline-block; padding: 8px 20px; background: var(--color-primary-light); color: var(--color-primary); border-radius: var(--radius); font-size: 14px; font-weight: 500; }
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

/* Crawl（在线更新法律） */
.crawl-section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 24px; }
.crawl-hint { font-size: 13px; color: var(--color-text-muted); }
.crawl-form { display: flex; flex-direction: column; gap: 14px; margin-top: 4px; }
.crawl-options { display: flex; gap: 20px; flex-wrap: wrap; font-size: 13px; color: var(--color-text-secondary); }
.checkbox-label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
.checkbox-label input { accent-color: var(--color-primary); }
.crawl-actions { display: flex; align-items: center; gap: 12px; }
.field-hint { color: var(--color-text-muted); font-size: 12px; }
.btn-crawl { padding: 10px 24px; background: var(--color-primary); color: #fff; border: none; border-radius: var(--radius); font-size: 15px; cursor: pointer; }
.btn-crawl:disabled { opacity: 0.5; cursor: not-allowed; }
.crawl-task { margin-top: 16px; border-top: 1px dashed var(--color-border); padding-top: 14px; display: flex; flex-direction: column; gap: 10px; }
.crawl-progress { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--color-text-secondary); }
.crawl-add { color: #059669; }
.crawl-upd { color: var(--color-primary); }
.crawl-fail { color: var(--color-error); }
.task-errors { display: flex; flex-direction: column; gap: 2px; }

/* Task progress */
.task-section { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 24px; margin-bottom: 24px; }
.task-section h3 { margin-bottom: 12px; font-size: 16px; }
.task-card { display: flex; flex-direction: column; gap: 10px; padding-bottom: 14px; border-bottom: 1px dashed var(--color-border); margin-bottom: 14px; }
.task-card:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.task-error { color: var(--color-error); font-size: 12px; }
.task-row { display: flex; justify-content: space-between; align-items: center; }
.task-id { font-family: monospace; font-size: 13px; }
.task-status { font-size: 12px; padding: 3px 10px; border-radius: 10px; background: var(--color-primary-light); color: var(--color-primary); }
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
.filter-select { padding: 6px 12px; border: 1px solid var(--color-border); border-radius: var(--radius); font-size: 13px; background: var(--color-surface-soft); color: var(--color-text); }
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
.type-badge.regulation { background: #D1FAE5; color: #065F46; }
.type-badge.judicial_interpretation,
.type-badge.interpretation, .type-badge.judicial { background: #DBEAFE; color: #1E40AF; }
.type-badge.local_regulation,
.type-badge.local { background: #FCE7F3; color: #9D174D; }
.type-badge.constitution { background: #FEF9C3; color: #854D0E; }
.type-badge.supervision { background: #E0E7FF; color: #3730A3; }
.type-badge.case { background: #FEF3C7; color: #92400E; }
/* 效力状态徽标 */
.status-badge { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 500; white-space: nowrap; }
.status-badge.active { background: #DCFCE7; color: #166534; }
.status-badge.repealed { background: #FEE2E2; color: #991B1B; }
.status-badge.revised { background: #E0E7FF; color: #3730A3; }
.status-badge.pending { background: #FEF3C7; color: #92400E; }

.actions-cell { white-space: nowrap; }
.btn-view { padding: 4px 12px; border: 1px solid var(--color-border); border-radius: var(--radius); background: var(--color-primary-light); color: var(--color-primary); cursor: pointer; font-size: 13px; margin-right: 6px; }
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
.chunk-index { font-weight: 600; color: var(--color-primary); font-size: 13px; }
.chunk-type { font-size: 12px; color: var(--color-text-muted); }
.chunk-content { margin: 0; padding: 12px; font-size: 14px; line-height: 1.8; color: var(--color-text); white-space: pre-wrap; word-break: break-word; font-family: var(--font-body); }
.view-more { text-align: center; padding: 14px 0 8px; font-size: 13px; color: var(--color-text-muted); }
.view-more-loading { display: inline-flex; align-items: center; gap: 6px; }
.view-more-loading::before {
  content: '';
  width: 12px; height: 12px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: kspin 0.6s linear infinite;
  display: inline-block;
}
@keyframes kspin { to { transform: rotate(360deg); } }
.view-more-done { color: var(--color-success); }

/* Modal */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 100; }
.modal { background: var(--color-surface); border-radius: var(--radius-lg); padding: 28px; max-width: 420px; width: 90%; box-shadow: 0 20px 60px rgba(0,0,0,0.15); }
.modal h3 { margin-bottom: 12px; font-size: 18px; }
.modal p { margin-bottom: 8px; color: var(--color-text-muted); font-size: 14px; }
.modal .warn { color: var(--color-error); font-size: 13px; margin-top: 12px; }
.modal-actions { display: flex; gap: 12px; justify-content: flex-end; margin-top: 20px; }
.btn-cancel { padding: 8px 20px; border: 1px solid var(--color-border); border-radius: var(--radius); background: var(--color-surface-soft); cursor: pointer; font-size: 14px; }
.btn-confirm-delete { padding: 8px 20px; border: none; border-radius: var(--radius); background: var(--color-error); color: #fff; cursor: pointer; font-size: 14px; }
.btn-confirm-delete:disabled { opacity: 0.5; }

/* 暗色模式下原生下拉列表可读性 */
::root[data-theme='dark'] .form-row select option,
::root[data-theme='dark'] .filter-select option {
  background: var(--color-surface-soft);
  color: var(--color-text);
}

/* ===== 深色模式覆盖：标签/徽标柔化为低饱和配色 ===== */
:root[data-theme='dark'] .type-badge,
:root[data-theme='dark'] .status-badge,
:root[data-theme='dark'] .task-status {
  background: var(--color-surface-hover);
  color: var(--color-text-secondary);
}
:root[data-theme='dark'] .type-badge.law { background: rgba(139, 92, 246, 0.18); color: #C4B5FD; }
:root[data-theme='dark'] .type-badge.regulation { background: rgba(16, 185, 129, 0.16); color: #6EE7B7; }
:root[data-theme='dark'] .type-badge.interpretation,
:root[data-theme='dark'] .type-badge.judicial { background: rgba(59, 130, 246, 0.18); color: #93C5FD; }
:root[data-theme='dark'] .type-badge.local_regulation,
:root[data-theme='dark'] .type-badge.local { background: rgba(236, 72, 153, 0.16); color: #F9A8D4; }
:root[data-theme='dark'] .type-badge.constitution { background: rgba(245, 158, 11, 0.16); color: #FCD34D; }
:root[data-theme='dark'] .type-badge.supervision { background: rgba(99, 102, 241, 0.18); color: #A5B4FC; }
:root[data-theme='dark'] .type-badge.case { background: rgba(245, 158, 11, 0.16); color: #FDE68A; }
:root[data-theme='dark'] .status-badge.active { background: rgba(16, 185, 129, 0.16); color: #6EE7B7; }
:root[data-theme='dark'] .status-badge.repealed { background: rgba(239, 68, 68, 0.16); color: #FCA5A5; }
:root[data-theme='dark'] .status-badge.revised { background: rgba(99, 102, 241, 0.18); color: #A5B4FC; }
:root[data-theme='dark'] .status-badge.pending { background: rgba(245, 158, 11, 0.16); color: #FCD34D; }
:root[data-theme='dark'] .task-status.done { background: rgba(16, 185, 129, 0.16); color: #6EE7B7; }
:root[data-theme='dark'] .task-status.failed { background: rgba(239, 68, 68, 0.16); color: #FCA5A5; }
:root[data-theme='dark'] .btn-view { background: rgba(99, 102, 241, 0.15); }
:root[data-theme='dark'] .btn-view:hover { background: rgba(99, 102, 241, 0.25); }
:root[data-theme='dark'] .btn-delete { border-color: rgba(248, 113, 113, 0.4); background: rgba(248, 113, 113, 0.12); color: #F87171; }
:root[data-theme='dark'] .btn-delete:hover { background: rgba(248, 113, 113, 0.22); }
:root[data-theme='dark'] .doc-table tr:hover { background: rgba(99, 102, 241, 0.08); }
</style>
