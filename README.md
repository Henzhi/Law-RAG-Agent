# Law-RAG-Agent

基于本地大语言模型的法律法规智能问答系统，集成 RAG 检索增强生成与 LangGraph Agent 多步骤任务调度。全程本地部署，无需联网。

---

## 技术栈

| 层次 | 技术 |
|:---|:---|
| LLM | Ollama + Qwen2.5:7b |
| Embedding | Ollama + bge-m3 (1024d) |
| Reranker | bge-reranker-v2-m3 (Cross-Encoder) |
| 向量索引 | FAISS IndexFlatIP |
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
│   ├── chunking/              # 法律文档解析 + 法条级切分
│   │   ├── parser.py           # 层次解析 (编→章→节→条) + 中文数字转换
│   │   └── chunker.py          # 智能切分 + 章级摘要
│   ├── embedding/             # 向量化
│   │   ├── embedder.py         # Ollama bge-m3 封装
│   │   └── vector_store.py     # FAISS 索引管理
│   ├── llm/                   # LLM 客户端
│   │   └── client.py           # Ollama Qwen2.5:7b 封装 + 流式
│   ├── rag/                   # RAG 引擎
│   │   ├── engine.py           # 问答管线 + Prompt 构建
│   │   ├── retriever.py        # 检索器抽象 (FAISS)
│   │   ├── hybrid_retriever.py # 混合检索 (向量+BM25)
│   │   ├── reranker.py         # Cross-Encoder 精排
│   │   └── adjacent_expander.py# 相邻条文上下文扩展
│   ├── agents/                # LangGraph Agent
│   │   ├── graph.py            # 6 节点状态图
│   │   └── tools.py            # 工具定义
│   ├── api/                   # FastAPI
│   │   ├── main.py             # 应用入口 + 日志配置
│   │   ├── routes.py           # API 路由 + 性能日志
│   │   ├── models.py           # Pydantic 模型
│   │   ├── auth.py             # JWT 认证
│   │   └── dependencies.py     # 依赖注入 (检索链)
│   └── config.py               # 全局配置
├── frontend/                  # Vue 3 前端
│   ├── src/views/              # 对话页 / 登录页
│   ├── src/stores/             # Pinia 状态管理
│   └── src/api/                # API 封装
├── scripts/
│   ├── build_index.py          # 构建 FAISS 索引
│   ├── demo.py                 # 演示脚本 (6 步流程)
│   ├── smoke_test.py           # 冒烟测试 (6 条路径)
│   ├── fill_eval_dataset.py    # 测试集自动填充
│   ├── eval_answer_quality.py  # 回答质量评测
│   ├── batch_eval.py           # 检索批量评测
│   └── generate_eval_dataset.py# 测试集生成
├── tests/
│   ├── test_chunking.py        # 44 用例 (解析+切分)
│   ├── test_rag_engine.py      # 33 用例 (闲聊/Prompt/条文)
│   ├── test_llm.py             # 17 用例 (Message/Config/构建)
│   ├── test_embedding.py       # 7 用例 (向量化/重试)
│   └── test_classify.py        # 30 用例 (意图分类)
├── data/
│   ├── vector_store/           # FAISS 索引 + BM25 语料
│   └── eval_dataset.json       # 131 条标注测试集
├── docs/                       # 文档
│   ├── technical_report.md     # 技术报告
│   ├── retrieval_eval.md       # 检索评测
│   ├── answer_quality.md       # 回答质量评测
│   ├── unit_test_report.md     # 单元测试报告
│   ├── smoke_test_report.md    # 冒烟测试报告
│   └── retrieval_noise_fix.md  # 检索噪声优化
├── docker/
│   └── init.sql                # PostgreSQL 初始化
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
uv sync

# 安装 Ollama 并拉取模型
ollama pull qwen2.5:7b
ollama pull bge-m3
```

### 2. 构建向量索引

```bash
uv run python scripts/build_index.py build
```

### 3. 启动服务

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

浏览器访问 `http://localhost:8000` 打开前端界面，或访问 `http://localhost:8000/docs` 查看 Swagger API 文档。

### 4. Docker 部署（可选）

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
| `EMBED_BATCH_SIZE` | `8` | 向量化批次大小 |
| `LLM_MODEL` | `qwen2.5:7b` | LLM 模型名 |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama 地址 |
| `LLM_TEMPERATURE` | `0.1` | 生成温度 |
| `LLM_TOP_P` | `0.9` | Nucleus 采样 |
| `LLM_MAX_TOKENS` | `2048` | 最大生成 token |
| `RETRIEVAL_TOP_K` | `5` | 检索返回条数 |
| `RETRIEVAL_HYBRID_ENABLED` | `false` | 混合检索开关（评测为负优化） |
| `RETRIEVAL_BM25_WEIGHT` | `0.0` | BM25 权重 |
| `RETRIEVAL_DROP_SUMMARY_CHUNKS` | `true` | 检索时过滤章级摘要噪声（消除 30+ 条无关条文召回） |
| `RERANK_ENABLED` | `true` | Reranker 精排开关 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker 模型 |
| `RERANK_RECALL_K` | `10` | 粗排候选数 |
| `RERANK_TOP_K` | `5` | 精排返回数 |
| `ADJACENT_ENABLED` | `true` | 相邻条文扩展 |
| `ADJACENT_WINDOW` | `2` | 扩展窗口 (±N) |
| `AGENT_ENABLED` | `false` | LangGraph Agent 开关（开启会额外发起改写+校验两次 LLM 调用，延迟上升；默认关以优先保证响应速度） |
| `INDEX_DIR` | `data/vector_store` | FAISS 索引路径 |
| `INDEX_NAME` | `law_index_bge` | 索引名称 |
| `JWT_SECRET` | (必填) | JWT 签名密钥 |
| `DATABASE_URL` | (可选) | PostgreSQL 连接串 |

---

## 核心功能

### RAG 检索流程

```
用户查询 → FAISS 向量检索 (bge-m3) → chunk_type 过滤
         → bge-reranker-v2-m3 精排 (Cross-Encoder)
         → 相邻条文扩展 (window=±2)
         → Prompt 拼接 → LLM 生成 → 答案
```

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

- 以「条」为最小切分单元，保持法律语义完整
- 短于 50 字的连续条文自动合并
- 为每章生成摘要 chunk，但检索时自动过滤（避免噪声）
- 每个 chunk 携带层次元数据 (法律名 → 编 → 章 → 节 → 条文范围)

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
| 最优配置 | 纯向量 (FAISS + bge-m3)，已移除章级摘要噪声 |

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
# 构建索引
uv run python scripts/build_index.py build

# 演示脚本 (6 步全流程)
uv run python scripts/demo.py

# 冒烟测试
uv run python scripts/smoke_test.py

# 测试集填充
uv run python scripts/fill_eval_dataset.py --resume

# 回答质量评测
uv run python scripts/eval_answer_quality.py

# 检索评测
uv run python scripts/batch_eval.py

# 单元测试
uv run pytest tests/ --ignore=tests/test_api.py -v
```

---

## 知识库

`LawData/` 目录包含 30 部中国法律原文，共 3449 条纯法条向量文档。涵盖：

刑法、民法典、宪法、行政处罚法、行政复议法、行政强制法、行政许可法、治安管理处罚法、道路交通安全法、食品安全法、环境保护法、劳动法、社会保险法、公司法、证券法、企业破产法、合伙企业法、个人独资企业法、信托法、票据法、专利法、商标法、著作权法、反不正当竞争法、监察法、立法法、公务员法、全国人大组织法、国务院组织法、行政诉讼法

数据来源：北大法宝公开法律数据库。

---

## 免责声明

本系统回答基于现行法律法规整理，仅供参考，不构成专业法律意见。涉及具体法律事务，请咨询执业律师。
