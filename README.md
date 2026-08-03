# Law-RAG-Agent

基于本地大语言模型的法律法规智能问答系统，集成 RAG 检索增强生成与 LangGraph Agent 多步骤任务调度。全程本地部署，无需联网。

---

## 技术栈

| 层次 | 技术 |
|:---|:---|
| LLM | Ollama + Qwen2.5:3b（流式） |
| Embedding | Ollama + bge-m3 (1024d) |
| Reranker | bge-reranker-v2-m3 (Cross-Encoder) |
| 向量索引 | pgvector (halfvec + HNSW) |
| Agent 框架 | LangGraph 1.2 |
| 后端 | Python 3.12 / FastAPI 0.115 / LangChain |
| 前端 | Vue 3 + Vite + Pinia |
| 认证 | JWT (python-jose) |
| 部署 | Docker + docker compose |

---

## 项目结构

```
Law-RAG-Agent/
├── src/
│   ├── knowledge/             # 知识处理（解析→切分→入库）
│   │   ├── ingestion/          # 切分管线（按「第X条」/按段落差异化）
│   │   │   └── pipeline.py     # 分类 + 切分 + 入库
│   │   ├── pgvector_store.py   # pgvector 存储/检索 (halfvec)
│   │   ├── doc_types.py        # 文档分类规范层 (doc_type/status)
│   │   ├── crawler/            # 法律爬虫（自动分类 + 增量去重）
│   │   └── ...
│   ├── embedding/             # 向量化 (Ollama bge-m3)
│   ├── llm/                   # LLM 客户端 (Ollama Qwen2.5:3b + 流式)
│   ├── rag/                   # RAG 引擎（检索链：粗排→精排→扩展→混合→路由）
│   │   ├── retriever.py         # 检索器抽象 + pgvector 粗排
│   │   ├── reranker.py          # Cross-Encoder 精排 (bge-reranker-v2-m3)
│   │   ├── adjacent_expander.py # 相邻条文上下文扩展
│   │   ├── article_router.py    # 条款号精确路由（"法名+第X条" 置顶）
│   │   ├── bm25_retriever.py    # BM25 关键词检索（法名入索引）
│   │   ├── hybrid_retriever.py  # 条件激活 rank-based 混合（加权 RRF）
│   │   ├── engine.py            # 问答管线 + Prompt 构建
│   │   └── ...
│   ├── agents/                # LangGraph Agent（意图识别→检索→生成→校验）
│   ├── api/                   # FastAPI（认证/路由/依赖注入检索链）
│   ├── memory/                # 会话记忆 + FAQ 缓存
│   └── config.py               # 全局配置
├── frontend/                  # Vue 3 前端
│   ├── src/views/              # 对话 / 登录 / 知识库
│   ├── src/stores/             # Pinia 状态管理
│   └── src/api/                # API 封装
├── scripts/                   # 业务/运维脚本（操作知识库）
│   ├── crawl.py                # 法律爬虫入口（直写 pgvector）
│   ├── ingest_lawdata.py       # LawData 批量导入
│   ├── rechunk_lawdata.py      # 存量重切分（按「第X条」）
│   ├── backfill_lawname.py     # 法律名称回填
│   └── rebuild_article_map.py  # article_map.json 重建
├── evaluation/                # 测试/评测（只读验证）
│   ├── scripts/                # 评测脚本（检索/回答质量/冒烟/测试集生成）
│   └── data/                   # 评测集（语义 100 条 / 法条级 339 条 / LexEval）
├── tests/                     # pytest 自动化测试（325 用例）
├── data/
│   └── vector_store/
│       └── article_map.json    # 条文映射表（相邻扩展运行时依赖）
└── docker/
    └── init.sql                # pgvector 表结构初始化
├── docs/                       # 文档
│   ├── technical_report.md     # 技术报告
│   ├── retrieval_eval.md       # 检索评测
│   ├── answer_quality.md       # 回答质量评测
│   ├── unit_test_report.md     # 单元测试报告
│   ├── smoke_test_report.md    # 冒烟测试报告
│   ├── retrieval_noise_fix.md  # 检索噪声优化
│   ├── adr-001-retrieval-config-alignment.md  # 检索配置对齐决策
│   └── adr-002-remove-chapter-summary.md      # 移除章级摘要决策
├── docker/
│   └── init.sql                # PostgreSQL 初始化
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## 快速开始

> 运行前需本地已具备：Python 3.12+、Node.js 18+、Ollama，并已拉取所需模型。

### 0. 模型准备

```bash
# 安装 Ollama 并拉取 LLM 与 Embedding 模型
# 说明: qwen2.5:7b 在 6GB 显存下 CPU 占用过高，生产使用 qwen2.5:3b（100% GPU）
ollama pull qwen2.5:3b
ollama pull bge-m3
# Reranker 由 sentence-transformers 本地加载 BAAI/bge-reranker-v2-m3
# （需提前缓存到本地 HF 目录；程序强制离线加载，不会联网下载）
```

### 1. 后端启动

```bash
# 0) 准备法律语料：将爬取并清洗好的法律 .txt 文件放入 LawData/ 目录

# 1) 安装 Python 依赖（uv 自动创建虚拟环境）
uv sync

# 2) 准备环境变量（按需修改 JWT_SECRET 等）
cp .env.example .env

# 3) 启动 PostgreSQL + pgvector（纯 PG 架构，无 FAISS）
docker compose up -d db

# 4) 导入法律数据（两种方式任选）
#    a) 知识库上传界面直接上传文档（按类型自动切分入库）
#    b) 命令行爬虫直写 pgvector
uv run python scripts/crawl.py --doc-type all --limit 50 --store pg

# 5) 启动 API 服务（默认 http://localhost:8000）
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. 前端构建

前端基于 Vite，`npm run build` 将产物输出到项目根目录 `static/`，由 FastAPI 直接静态托管：

```bash
cd frontend
npm install        # 安装依赖
npm run build      # 构建产物输出到 ../static
```

构建完成后访问 `http://localhost:8000` 即可打开对话界面；`http://localhost:8000/docs` 为 Swagger API 文档。

> 开发模式（热更新，免每次 build）：保持步骤 1 的后端运行，另开终端执行
> ```bash
> cd frontend && npm run dev   # http://localhost:3000，已配置 /api 代理转发到 8000
> ```

### 3. Docker 部署（可选）

```bash
docker compose up -d
```

---

## API 接口

| 方法 | 路径 | 说明 | 认证 |
|:---|:---|:---|:---:|
| `GET` | `/api/health` | 健康检查（状态/索引/文档数） | — |
| `POST` | `/api/chat` | 法律问答（完整答案 + 引用来源） | Bearer |
| `POST` | `/api/chat/stream` | 流式问答（SSE 逐字输出） | Bearer |
| `POST` | `/api/auth/register` | 用户注册 | — |
| `POST` | `/api/auth/login` | 用户登录（返回 JWT） | — |
| `POST` | `/api/crawl` | 触发爬取任务（国家法律法规数据库，增量更新） | — |
| `GET` | `/api/crawl/status/{task_id}` | 查询爬取任务状态与结果 | — |
| `GET` | `/api/crawl/types` | 列出支持的爬取类型 | — |

### 请求示例

```bash
# 健康检查
curl http://localhost:8000/api/health

# 单次问答
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "治安管理处罚有哪几种", "top_k": 5}'

# 流式问答
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "行政拘留最长多久", "top_k": 3}'

# 多轮对话 (带 history)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "那防卫过当呢",
    "history": [
      {"role": "user", "content": "正当防卫怎么认定"},
      {"role": "assistant", "content": "根据《刑法》第二十条..."}
    ],
    "top_k": 3
  }'
```

### 响应格式

```json
{
  "query": "治安管理处罚有哪几种",
  "answer": "根据《中华人民共和国治安管理处罚法(2025修订)》第十条，治安管理处罚的种类分为：（一）警告；（二）罚款；（三）行政拘留；（四）吊销公安机关发放的许可证。\n\n⚠️ 以上内容基于现行法律法规整理，仅供参考，不构成专业法律意见。如涉及具体法律事务，请咨询执业律师。",
  "sources": [
    {
      "law_name": "中华人民共和国治安管理处罚法(2025修订)",
      "chapter": "第二章 处罚的种类和适用",
      "article_range": "第十条",
      "citation": "治安管理处罚法(2025修订) · 第十条",
      "score": 0.9521
    }
  ],
  "is_casual": false
}
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|:---|:---|:---|
| `EMBED_MODEL` | `bge-m3` | Embedding 模型名 |
| `EMBED_BASE_URL` | `http://localhost:11434` | Ollama 地址 |
| `EMBED_BATCH_SIZE` | `32` | 向量化批次大小（Ollama 受限环境可调小至 8） |
| `LLM_MODEL` | `qwen2.5:3b` | LLM 模型名 |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama 地址 |
| `LLM_TEMPERATURE` | `0.1` | 生成温度 |
| `LLM_TOP_P` | `0.9` | Nucleus 采样 |
| `LLM_MAX_TOKENS` | `2048` | 最大生成 token |
| `RETRIEVAL_TOP_K` | `5` | 检索返回条数 |
| `RETRIEVAL_DROP_SUMMARY_CHUNKS` | `true` | 检索时过滤章级摘要噪声（消除 30+ 条无关条文召回） |
| `ADJACENT_ENABLED` | `true` | 相邻条文扩展开关 |
| `ADJACENT_WINDOW` | `1` | 相邻扩展窗口（±N 条，曾用 3 导致引用噪声，改 1） |
| `RETRIEVAL_SIM_THRESHOLD` | `0.0` | 向量相似度召回闸门（bge-m3 归一化内积，范围约 [-1,1]）；>0 时低于阈值的候选被丢弃，过滤后无候选则回退保留；建议 0.3~0.5 |
| `RERANK_ENABLED` | `true` | Reranker 精排开关 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker 模型 |
| `RERANK_RECALL_K` | `15` | 粗排候选数 |
| `RERANK_TOP_K` | `15` | 精排返回数 |
| `ADJACENT_ENABLED` | `true` | 相邻条文扩展（放在 Reranker 之前丰富候选） |
| `ADJACENT_WINDOW` | `1` | 扩展窗口 (±N) |
| `AGENT_ENABLED` | `false` | LangGraph Agent 开关（含查询改写+答案校验+重试）；开启会增加 rewrite/validate 两次 LLM 调用、延迟上升，追求最高质量时设 `true`（`.env.example` 已开启） |
| `AGENT_MAX_RETRIES` | `1` | 答案校验失败时的最大重试次数 |
| `JWT_SECRET` | (必填) | JWT 签名密钥 |
| `PG_ENABLED` | `true` | 纯 PG 架构：必须为 true（v0.6 起无 FAISS 回退） |
| `PG_CONN` | `postgresql://lawrag:lawrag123@localhost:5432/lawrag` | pgvector 连接串 |
| `HYBRID_ENABLED` | `true` | BM25 条件混合检索（仅法名/条款查询激活） |
| `HYBRID_RRF_K` | `60` | RRF 融合常数（只看排名不碰分数） |
| `HYBRID_BM25_WEIGHT` | `3.0` | BM25 路融合权重（向量路=1.0） |

---

## 核心功能

### RAG 检索流程

```
用户查询
  → 条款号路由: 含"法名+第X条"时精确置顶 (ArticleRouter)
  → pgvector 向量检索 (bge-m3, halfvec, 精确扫描)
  → 相邻条文扩展 (window=±1)
  → bge-reranker-v2-m3 精排 (Cross-Encoder, Top 15)
  → 条件 BM25 混合: 查询含法名/条款号时按排名加权融合 (加权 RRF)
  → Prompt 拼接 → LLM 生成 → 答案
```

检索质量（2026-08 评测，法条级 339 条测试集）：Hit@1 68.1% / Hit@5 86.1% / Hit@10 91.7%

### Agent 工作流 (LangGraph)

```
intent (意图识别)
  ├─ casual → casual_reply (闲聊直接回复)
  └─ legal  → rewrite (查询改写)
              → retrieve (向量检索)
              → generate (生成回答)
              → validate (答案校验)
                  ├─ pass → END
                  └─ fail → generate (重试)
```

### 切分策略

按文档类型差异化切分（`src/knowledge/ingestion/pipeline.py`）：
- **条文体**（law / regulation / constitution / supervision）：以「第X条」为天然边界，每个条文独立成块（短条文不合并）；超长条文按句号拆分且续块保留条号前缀，保证引用可追溯
- **非条文体**（judicial_interpretation / case）：按自然段切分，不硬拆条文（保持案件事实/解释脉络连续）
- 每个 chunk 携带层次元数据（法律名 → 条文范围），供检索结果「引用条文」展示

### Prompt 设计

- 角色设定：专业法律助手
- 5 条约束：引用法条/不编造/诚实/简洁/免责声明
- 带 Few-shot 示例（治安处罚、酒驾处罚等）
- 末尾自动附加免责声明

---

## 评测结果

### 检索质量

| 指标 | 数值 |
|:---|:---:|
| Recall@5 | **73.00%** |
| Recall@10 | 81.00% |
| MRR | 0.6113 |
| 最优配置 | 纯向量 (pgvector + bge-m3)，已移除章级摘要噪声 |

详见: `docs/retrieval_eval.md`

### 回答质量

| 指标 | 数值 |
|:---|:---:|
| 综合评分 | **0.890** |
| 优秀率 (≥0.8) | 75.6% |
| 法律名称命中率 | 95.4% |
| 法条号命中率 | 77.0% |
| 检索失败率 | 1.5% |
| 真实幻觉率 | **0%** |

详见: `docs/answer_quality.md`

### 测试覆盖

| 指标 | 数值 |
|:---|:---:|
| 单元测试 | 174 用例 / 全通过 |
| 冒烟测试 | 14/16 通过 |

---

## 命令行工具

```bash
# 冒烟测试
uv run python evaluation/scripts/smoke_test.py

# 检索质量评测（100 条语义评测集）
uv run python evaluation/scripts/eval_retrieval.py

# 法条级评测集生成（按知识库实际条文）
uv run python evaluation/scripts/generate_article_queries.py --per-law 3 --max-laws 60

# 回答质量评测
uv run python evaluation/scripts/eval_answer_quality.py

# 单元测试
uv run pytest tests/ --ignore=tests/test_api.py -v
```

---

## 知识库

知识库基于 LawData 爬虫数据入库，当前规模 **931 篇文档 / 51348 chunks**（按「第X条」重切分后），doc_type 分布：regulation 594 / law 295 / judicial_interpretation 40 / case 2。覆盖刑法、民法典、宪法、行政处罚法、行政复议法、治安管理处罚法、食品安全法、劳动法、社会保险法、公司法、证券法、专利法、商标法、著作权法等主要法律。

（旧版本地 30 部原文测试数据已由 LawData 全量数据取代。）

数据来源：北大法宝公开法律数据库。

---

## 爬取功能（增量更新）

内置「国家法律法规数据库」（全国人大官方，flk.npc.gov.cn）爬虫，可将法律正文抓取、清洗后**增量**落地到 `LawData/<子目录>/`，避免重复下载已存在的法律。

- **数据源**：全国人大官方，免费、权威、合规风险低（请仅用于学习 / 研究，并控制请求频率）
- **支持类型**：`law`(法律法规)、`regulation`(行政法规)、`judicial`(司法解释)、`local`(地方性法规)、`constitution`(宪法)、`supervision`(监察法规)、`all`(全部)；`case`(案例 / 裁判文书) 该数据源不提供
- **增量策略**：每个子目录维护 `.crawl_manifest.json`，按文档 id 去重；`force=true` 可强制重爬
- **落库方式**（`--store` / API 的 `store` 字段）：
  - `pg`（默认）：**直接写入 PostgreSQL + pgvector**（需 `PG_ENABLED=true` 且已 `docker compose up -d`），无需落盘即可被检索
  - `txt`：落地到 `LawData/<子目录>/*.txt`（原始文本存档）
  - `both`：两者都做
  - pg 模式下的增量按「标题」去重（已入库则跳过，`force=true` 会删除旧文档重建）

### API 触发

```bash
# 1) 提交爬取任务（后台执行，返回 task_id）
curl -X POST http://localhost:8000/api/crawl \
  -H "Content-Type: application/json" \
  -d '{"source":"npc","doc_type":"law","keyword":"刑法","limit":10,"force":false}'

# 2) 轮询进度与结果
curl http://localhost:8000/api/crawl/status/{task_id}

# 3) 查看支持的类型
curl http://localhost:8000/api/crawl/types
```

### 命令行触发（可选）

```bash
uv run python -c "from src.knowledge.crawler import NpcLawCrawler; \
  print(NpcLawCrawler().crawl(doc_type='law', keyword='刑法', limit=10))"
```

### 爬取后重建索引

使用 `store=pg`（默认）时，爬取的文档**直接写入 pgvector，无需重建索引**即可被检索。

`/api/crawl` 请求体中的 `"rebuild": true` 会在任务完成后由服务自动对 pgvector 做一次 HNSW 全量重建（增量插入已生效，一般无需开启）。

---

## 免责声明

本系统回答基于现行法律法规整理，仅供参考，不构成专业法律意见。涉及具体法律事务，请咨询执业律师。
