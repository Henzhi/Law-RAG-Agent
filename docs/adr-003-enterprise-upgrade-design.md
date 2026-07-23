# ADR-003: Law-RAG-Agent 企业级升级技术设计

> **状态**: 实施中
> **日期**: 2026-07-23
> **决策人**: 单人开发 + AI 辅助
> **前置阅读**: README.md、adr-001、adr-002
>
> ### 实施日志
> | 日期 | 步骤 | 内容 |
> |------|------|------|
> | 07-23 | 1a | LLM 后端抽象层 — base.py / ollama_backend.py / openai_backend.py / factory.py + 19 测试 ✅ |
> | 07-23 | 1b | Embedding 后端抽象层 — base.py / ollama_embedder.py / openai_embedder.py / factory.py + 24 测试 ✅ |
| 07-23 | 1c | config.py 统一配置 + adapter 适配器 + dependencies.py 接入 ✅ |
| 07-23 | 1d | 全面审查修复 — Bug1: LLMMessage兼容 + Bug2: 注释过期 + 6测试 ✅ |
| 07-23 | 1e | 二次审查 — Bug3: base_url污染(Ollama URL误入OpenAI后端) 已修复 ✅ |
| 07-23 | 2  | pgvector 完全替代 FAISS — 进行中 |

---

## 1. 升级目标

将当前 MVP 原型升级为可交付的私有化部署产品。

| 维度 | 当前 (v0.1) | 目标 (v0.5) |
|------|-------------|-------------|
| 记忆系统 | 截断最近 N 条历史消息 | 对话记忆 + FAQ 语义缓存 |
| 模型支持 | Ollama only | Ollama + OpenAI 兼容双后端 |
| 向量存储 | FAISS 为主 | pgvector 为主 |
| 知识库 | 30 部法律，静态 JSON | 多类型知识 + 文档上传 + 增量索引 |
| 意图识别 | 闲聊/法律 二分类 | 法律条文查询/案例参考/其他 三分类 |
| 可观测性 | logging 基础输出 | 结构化日志 + 检索质量追踪 |
| 部署 | docker-compose 单机 | docker-compose 单机（架构支持未来 K8s） |
| 缓存 | 无 | Redis (语义 FAQ + Embedding 缓存) |

---

## 2. 技术架构

### 2.1 整体架构

```
┌────────────────────────────────────────────────────────────┐
│                    Vue 3 前端 (Vite)                        │
│  页面：问答 | 知识库管理 | 对话历史                          │
└────────────────────────┬───────────────────────────────────┘
                         │ HTTP + SSE
┌────────────────────────▼───────────────────────────────────┐
│              FastAPI 应用服务 (单进程)                        │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │ LLM 后端 │ │Embedding │ │  记忆系统 │ │ 知识库   │     │
│  │ 抽象层   │ │  抽象层   │ │  管理器   │ │ 管理器   │     │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘     │
│       │            │            │            │             │
│  ┌────▼────────────▼────────────▼────────────▼─────┐      │
│  │              LangGraph Agent (6→8 节点)           │      │
│  │  intent → memory_retrieve → rewrite → retrieve→ │      │
│  │            generate → validate → END             │      │
│  └───────────────────────┬─────────────────────────┘      │
└──────────────────────────┼────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐
│  PostgreSQL    │  │    Redis      │  │  Ollama/API   │
│  + pgvector    │  │  缓存+会话    │  │  LLM+Embed    │
└───────────────┘  └───────────────┘  └───────────────┘
```

### 2.2 新增模块清单

```
src/
├── llm/                        # [重构] LLM 多后端抽象
│   ├── base.py                 # 抽象基类 LLMBackend
│   ├── ollama_backend.py       # Ollama 实现
│   ├── openai_backend.py       # OpenAI 兼容 API 实现
│   └── factory.py              # 工厂函数 + Token 预算管理
├── embedding/                  # [重构] Embedding 多后端抽象
│   ├── base.py                 # 抽象基类 EmbeddingBackend
│   ├── ollama_embedder.py      # Ollama bge-m3
│   ├── openai_embedder.py      # OpenAI text-embedding-3
│   └── factory.py              # 工厂函数
├── memory/                     # [新增] 记忆系统
│   ├── __init__.py
│   ├── manager.py              # 记忆协调器
│   ├── conversation.py         # 对话记忆 (摘要 + 检索)
│   ├── faq_cache.py            # FAQ 语义缓存
│   ├── summarizer.py           # 对话摘要生成器
│   └── token_budget.py         # 上下文窗口 Token 预算管理
├── knowledge/                  # [新增] 知识库管理
│   ├── __init__.py
│   ├── models.py               # 统一文档 schema
│   ├── document_store.py       # PostgreSQL 文档存储
│   ├── ingestion/              # 文档解析管道
│   │   ├── pipeline.py         # 处理管道
│   │   ├── pdf_parser.py       # PDF 解析
│   │   ├── docx_parser.py      # Word 解析
│   │   └── text_cleaner.py     # 文本清洗
│   └── index_manager.py        # 增量索引管理
├── rag/                        # [扩展] RAG 引擎
│   ├── engine.py               # [修改] 集成记忆+Token预算
│   ├── retriever.py            # [重写] pgvector 主检索器
│   ├── hybrid_retriever.py     # [保留] 混合检索(备选)
│   ├── reranker.py             # [保留] Cross-Encoder 精排
│   ├── adjacent_expander.py    # [保留] 相邻条文扩展
│   └── intent.py               # [扩展] 三分类意图识别
├── agents/                     # [修改] Agent 工作流
│   ├── graph.py                # [修改] 增加 memory_retrieve 节点
│   └── tools.py                # [扩展] 新增知识库查询工具
├── api/                        # [扩展] 新增接口
│   ├── routes.py               # [修改] 新增知识库/记忆路由
│   ├── models.py               # [修改] 新增请求/响应模型
│   └── conversation_store.py   # [保留] 对话持久化
├── observability/              # [新增] 可观测性
│   ├── logger.py               # structlog 结构化日志
│   ├── trace.py                # 请求追踪
│   └── query_log.py            # 检索质量日志表
└── config.py                   # [修改] 新增配置项
```

---

## 3. 详细模块设计

### 3.1 多模型抽象层

#### 3.1.1 LLM 后端

```python
# src/llm/base.py - 抽象接口
from abc import ABC, abstractmethod
from typing import AsyncIterator

class LLMBackend(ABC):
    """LLM 后端抽象基类"""

    @abstractmethod
    async def generate(self, messages: list[dict], **kwargs) -> str:
        """同步生成回答"""
        ...

    @abstractmethod
    async def stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """流式生成"""
        ...

    @abstractmethod
    def get_context_window(self) -> int:
        """返回该模型的上下文窗口大小 (tokens)"""
        ...
```

#### 3.1.2 配置驱动切换

```env
# .env
LLM_BACKEND=ollama           # ollama | openai | vllm
LLM_MODEL=qwen2.5:7b         # 模型名

# Ollama 配置
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI 兼容配置 (DeepSeek / 通义千问 / vLLM / OpenAI)
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

#### 3.1.3 Embedding 后端

```python
# src/embedding/base.py
from abc import ABC, abstractmethod

class EmbeddingBackend(ABC):
    """Embedding 后端抽象基类"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化"""
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """返回向量维度"""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型标识，用于 pgvector 维度隔离"""
        ...
```

---

### 3.2 记忆系统 MVP

#### 3.2.1 对话记忆

```
生命周期：跨会话，30 天

存储表 (PostgreSQL + pgvector):
┌─────────────────────────────────────────────────────┐
│ conversation_memories                                │
├─────────────────────────────────────────────────────┤
│ id              UUID PRIMARY KEY                    │
│ user_id         VARCHAR(128) NOT NULL               │
│ session_id      VARCHAR(128) NOT NULL               │
│ summary         TEXT              -- LLM 生成的摘要  │
│ summary_embed   VECTOR(1024)      -- 摘要向量        │
│ entities        JSONB             -- 关键实体         │
│   {case_type, laws_involved, parties, key_facts}    │
│ message_count   INT               -- 对话轮数         │
│ created_at      TIMESTAMPTZ                         │
│ expires_at      TIMESTAMPTZ       -- TTL: created+30d│
└─────────────────────────────────────────────────────┘

触发条件:
- 对话轮数 > 6 轮时触发异步摘要生成
- 摘要结构: {case_type, laws, key_entities, conclusion, open_questions}

检索流程:
1. 用户新问题 → embedding
2. pgvector 检索: WHERE user_id=xxx ORDER BY embedding <-> query LIMIT 3
3. 时间衰减: 越新的摘要权重越高
4. 拼入 Prompt [历史参考] 段
```

#### 3.2.2 FAQ 语义缓存

```
生命周期：TTL 7 天，关联法律变更时级联失效

存储表 (PostgreSQL + pgvector):
┌─────────────────────────────────────────────────────┐
│ faq_cache                                           │
├─────────────────────────────────────────────────────┤
│ id              UUID PRIMARY KEY                    │
│ question        TEXT NOT NULL                       │
│ question_embed  VECTOR(1024)                        │
│ answer          TEXT NOT NULL                       │
│ sources         JSONB             -- 引用来源         │
│ related_laws    TEXT[]            -- 关联法律ID列表   │
│ confidence      FLOAT             -- 置信度           │
│ hit_count       INT DEFAULT 1     -- 命中次数         │
│ created_at      TIMESTAMPTZ                         │
│ expires_at      TIMESTAMPTZ                         │
│ status          VARCHAR(20) DEFAULT 'active'        │
│   -- active | expired | invalidated                 │
└─────────────────────────────────────────────────────┘

命中条件:
- cosine_similarity > 0.95
- status = 'active'
- expires_at > NOW()

法律修订失效:
- 法律 V2 生效 → 级联标记 related_laws 包含该 ID 的所有缓存
- status: active → invalidated
```

#### 3.2.3 Token 预算管理

```python
# src/memory/token_budget.py

class TokenBudget:
    """上下文窗口 Token 预算管理器"""

    # 默认分配比例 (28K 窗口为例)
    ALLOCATION = {
        "system_prompt":    {"tokens": 800,   "priority": "required"},
        "memory_context":   {"tokens": 1500,  "priority": "high"},
        "retrieval_docs":   {"tokens": 8000,  "priority": "highest"},
        "chat_history":     {"tokens": 3000,  "priority": "medium"},
        "user_query":       {"tokens": 500,   "priority": "required"},
        "output_reserve":   {"tokens": 12000, "priority": "required"},
    }

    def __init__(self, context_window: int):
        self.total = context_window - 2000  # 预留 2K 安全边界

    def compute(self, query_complexity: str) -> dict:
        """根据查询复杂度动态调整分配"""
        ...

    def assemble(self, components: dict) -> str:
        """组装最终 Prompt，确保不超窗口"""
        ...
```

#### 3.2.4 LangGraph 工作流更新

```
当前 (6 节点):
intent → rewrite → retrieve → generate → validate

升级后 (8 节点):
intent → memory_retrieve → rewrite → retrieve → generate → validate
            ↑ 新增节点                              ↓
            └── 检索历史摘要                        ├─ PASS → END
                + FAQ 缓存检查                      └─ FAIL → generate (重试)
```

### 3.3 多集合并行检索

> **决策**: Phase 1 不做 LangGraph 多 Agent，在 retrieve 节点内部实现 asyncio.gather 并行检索。

#### 3.3.1 设计原则

```
多 Agent 与否的权衡：

┌──────────────────────────────────────────────────────────┐
│                      多 Agent 价值分析                     │
├──────────────┬──────────────┬──────────────┬─────────────┤
│  方案          │ 复杂度        │ 简单查询延迟  │ 复杂查询延迟  │
├──────────────┼──────────────┼──────────────┼─────────────┤
│ 单Agent串行    │ 低 (当前)     │ 快 2s         │ 慢 7s        │
│ 多Collection   │ 中 (推荐)     │ 一样快 2s      │ 快 4s        │
│ 并行检索        │              │              │             │
│ LangGraph      │ 高            │ 更慢 2.5s      │ 最快 3.5s    │
│ 多Agent        │              │              │             │
└──────────────┴──────────────┴──────────────┴─────────────┘

结论: 多集合并行检索 = 80% 多Agent价值 + 0% 多Agent复杂度
```

#### 3.3.2 架构

```python
# src/rag/retriever.py — 多集合并行检索

import asyncio

class MultiCollectionRetriever:
    """并行检索多个知识集合"""

    def __init__(self):
        self.collections = {}  # Phase 1 根据实际数据注册

    async def retrieve(self, query: str, intent: str, top_k: int):
        # 根据意图决定需要检索哪些集合
        collections = self._select_collections(intent)

        # 并行检索（asyncio.gather）
        tasks = [
            retriever.search(query, top_k)
            for name, retriever in collections.items()
        ]
        all_results = await asyncio.gather(*tasks)

        # 合并 + 统一精排
        return self._merge_and_rerank(all_results)

    def _select_collections(self, intent: str):
        """意图 → 检索集合映射"""
        if intent == "law_lookup":
            return {"law": self.collections["law"]}
        elif intent == "case_query":
            return {"case": self.collections["case"]}
        elif intent == "comprehensive":
            return self.collections  # 全部并行
        else:
            return {"law": self.collections["law"]}  # 默认
```

#### 3.3.3 多 Agent 预留

```python
# src/agents/graph.py — Phase 1 保持不变，架构预留

def build_graph():
    builder = StateGraph(AgentState)

    # retrieve 节点 — Phase 1: 单节点统一处理
    builder.add_node("retrieve", unified_retrieve_node)

    # Phase 2 替换方案 (不实现):
    # builder.add_node("retrieve_law", law_retrieve_node)
    # builder.add_node("retrieve_case", case_retrieve_node)
    # builder.add_node("retrieve_interpretation", interp_retrieve_node)
    # builder.add_node("merge_retrieval", merge_node)
    #
    # 用 LangGraph Send() 并行分发:
    #   retrieve → [Send("retrieve_law"), Send("retrieve_case")]
    #            → merge_retrieval
```

#### 3.3.4 触发条件（何时升级到多 Agent）

```
满足任意 2 项:
☐ 知识库扩展到 3 类以上 (法条+案例+司法解释+地方性法规)
☐ 复杂查询占比 > 30%
☐ 单次查询延迟 > 8s
☐ 不同知识类型需要完全不同的检索策略 (不同 embedding/reranker)
☐ 有真实反馈数据证明多 Agent 能解决实际问题

预计触发: Phase 2 中后期 (知识库 5万+ 向量文档时)
```

---

### 3.4 知识库管理系统

#### 3.4.1 统一文档 Schema

```json
{
  "doc_id": "law_0001",
  "doc_type": "law|interpretation|case|regulation",
  "title": "中华人民共和国刑法",
  "source": "全国人大",
  "effective_date": "1997-10-01",
  "version": 1,
  "status": "active|superseded|draft",
  "superseded_by": null,
  "chunks": [
    {
      "chunk_id": "law_0001_art_001",
      "chunk_type": "article|judgment|summary|guideline",
      "content": "为了惩罚犯罪，保护人民...",
      "embedding_model": "bge-m3",
      "metadata": {
        "chapter": "第一编 总则",
        "article_no": "第一条",
        "related_laws": [],
        "related_cases": []
      }
    }
  ]
}
```

#### 3.4.2 数据库表设计

```sql
-- 文档表
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_type VARCHAR(20) NOT NULL,  -- law|interpretation|case|regulation
    title VARCHAR(500) NOT NULL,
    source VARCHAR(500),
    effective_date DATE,
    version INT DEFAULT 1,
    status VARCHAR(20) DEFAULT 'active',  -- active|superseded|draft
    superseded_by UUID REFERENCES documents(id),
    original_filename VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 文档块表 (pgvector)
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_type VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    embedding_model VARCHAR(50) NOT NULL,
    embedding VECTOR(3072),             -- 按最大维度预留
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 检索质量日志
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    user_id VARCHAR(128),
    query TEXT NOT NULL,
    intent VARCHAR(20),
    retrieved_count INT,
    reranked_count INT,
    faq_cache_hit BOOLEAN DEFAULT FALSE,
    memory_docs_used INT DEFAULT 0,
    llm_tokens_used INT,
    total_latency_ms INT,
    stage_timings JSONB,               -- {intent_ms, retrieve_ms, ...}
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

### 3.5 意图识别增强

```
当前: 闲聊 / 法律 (二分类)

升级后 (三分类):

用户输入 → intent 节点
         ├── casual        → 闲聊
         ├── law_lookup    → 法律条文检索 (主RAG)
         ├── case_query    → 案例检索 (语义匹配)
         └── other         → 超出知识库范围

每种意图对应不同检索策略:
- law_lookup: 精排 Top-5 法条 + 相邻条文扩展
- case_query: 语义检索 Top-3 案例 + 关联法条
- other: 不检索，提示超出范围

Prompt 模板也随意图动态切换
```

---

### 3.6 幻觉防御

```
四层校验:

Layer 1 (已有): 法条存在性检查 (validate 节点)
Layer 2 (已有): 引用精确性检查 (validate 节点)
Layer 3 (新增): 检索置信度检查
  - 检索结果 max_similarity < 0.7 → 视为低置信度
  - 回复"该问题超出当前知识库范围"，不强行生成
Layer 4 (强化): 免责声明
  - 自动在输出末尾追加
  - "以上内容基于现行法律法规整理，仅供参考，不构成法律意见"

内容安全（新增）:
- 输入过滤: 检测 "忽略指令"、"system prompt"、敏感词
- 输出过滤: 涉黄涉政关键词检测
- 攻击/违规 → 统一回复 "该问题不在我的服务范围内"
```

---

### 3.7 可观测性基础

```
每个请求记录：
{
  "request_id": "uuid",
  "user_id": "xxx",
  "query": "工伤认定标准...",
  "intent": "law_lookup",
  "retrieved_docs": 15,
  "reranked_docs": 5,
  "faq_cache_hit": false,
  "memory_docs_used": 2,
  "llm_tokens_used": 1240,
  "total_latency_ms": 3200,
  "stage_timings": {
    "intent_ms": 200,
    "memory_ms": 150,
    "rewrite_ms": 300,
    "retrieve_ms": 150,
    "rerank_ms": 300,
    "generate_ms": 2100
  }
}

存储: PostgreSQL query_logs 表
用途: 检索质量分析、性能瓶颈定位、高频问题发现

未来可接入: OpenTelemetry + Grafana
```

---

### 3.8 pgvector 性能防护

```
索引策略:
- document_chunks: IVFFlat (精度优先，list=sqrt(行数))
- faq_cache: HNSW (速度优先，m=16, ef_construction=200)
- conversation_memories: 精确搜索 (数据量小，无需近似索引)

维度优化:
- 使用 pgvector halfvec (半精度 float16)
  → 存储减半，检索速度提升 30-40%
  → 精度损失 < 0.1%，法律场景可接受

Embedding 模型隔离:
- embedding_model 字段标记每条向量的模型
- 查询时 WHERE embedding_model = current_model
- 切换模型 = 新建索引 + 后台并行重建 + 原子切换
```

---

## 4. 实施计划

### 4.1 Phase 1 任务清单（8 步，预计 4-6 周）

| # | 任务 | 产出 | 预计 |
|---|------|------|------|
| 1 | 项目结构重构 + 多模型抽象层 | 可切换 Ollama/OpenAI | 2-3 天 |
| 2 | pgvector 完全替代 FAISS | 增量索引 + 预留维度隔离 | 2-3 天 |
| 3 | 对话记忆层 | 跨会话摘要检索 | 2-3 天 |
| 4 | FAQ 语义缓存 | 高频问题直接响应 | 1-2 天 |
| 5 | 文档上传 + 解析管道 | PDF/Word → 向量入库 | 2-3 天 |
| 6 | 知识库扩展 + 意图识别增强 | 三分类 + 多类型知识 | 1-2 天 |
| 7 | Token 预算 + 记忆时序修复 + 幻觉防御 | 不爆窗口 + 时序正确 | 2-3 天 |
| 8 | 可观测性 + 前端新页面 | 日志 + KB管理 + 历史 | 2-3 天 |

### 4.2 编码顺序依赖

```
步骤 1 (多模型抽象层)
  └─→ 步骤 2 (pgvector) ──→ 步骤 5 (文档上传)
  └─→ 步骤 6 (意图识别)
  └─→ 步骤 3 (对话记忆) ──→ 步骤 4 (FAQ缓存)
       └─→ 步骤 7 (Token预算+时序修复) ──→ 步骤 8 (日志+前端)

步骤 1 是基础，必须先做。
步骤 2-4 可适度并行。
步骤 7 是集成步骤，需要 3+4+6 完成后再做。
```

---

## 5. 风险登记册

按致命程度排序：

| # | 风险 | 等级 | 防御措施 | 阶段 |
|---|------|------|----------|------|
| R1 | 检索失败后 LLM 自由发挥，给出虚构法律建议 | 🔴致命 | 设最低相似度阈值 0.7，低于阈值拒绝回答 | 步骤 7 |
| R2 | FAQ 缓存返回已废止法律的旧答案 | 🔴致命 | 缓存 TTL 绑定法律版本，修法时级联失效 | 步骤 4 |
| R3 | Prompt 注入 / 越狱攻击 | 🔴致命 | 输入输出关键词过滤，违规统一拒绝 | 步骤 7 |
| R4 | Embedding 模型切换导致全量重建 | 🟡严重 | 表设计预留 embedding_model 字段隔离 | 步骤 2 |
| R5 | 流式输出 + 记忆检索时序冲突 | 🟡严重 | Graph 中 memory_retrieve 放在 retrieve 之前 | 步骤 7 |
| R6 | 法律修订时新旧条文并存 | 🟡严重 | 文档版本管理，status 字段标记，原子切换 | 步骤 5 |
| R7 | PDF 解析准确率不足 | 🟡严重 | 文本清洗管道 + 上传后预览确认 | 步骤 5 |
| R8 | pgvector 大数据量性能下降 | 🟢一般 | halfvec + 索引参数调优 + 预留 Qdrant 接入 | 步骤 2 |
| R9 | LangGraph Checkpoint 数据膨胀 | 🟢一般 | 定时清理过期 checkpoint | 步骤 7 |
| R10 | 对话数据隐私泄露 | 🟡严重 | 脱敏 + 数据库加密 + 删除接口 | 步骤 8 |

---

## 6. 部署与环境

### 6.1 开发环境

```
方案 B (开发用):
  LLM: DeepSeek / 通义千问 API
  Embedding: Ollama bge-m3 本地 GPU
  VRAM 需求: ~2.4GB (仅 Embedding)
  RAM 需求: ~12-14GB (PG + Redis + FastAPI + Ollama)

理由: RTX 4050 (6GB) 无法同时跑 LLM + Embedding
      API 方案效果远好于 7B 量化模型
```

### 6.2 客户部署配置

```
模式 1: 纯本地部署 (客户有 GPU)
  最低: RTX 3060 12GB + 32GB RAM
  推荐: RTX 4070 12GB + 64GB RAM
  组件: Ollama (LLM + Embedding) + PG + Redis + App

模式 2: API 混合部署 (客户用 API)
  最低: 16GB RAM (无 GPU 要求)
  组件: PG + Redis + App (+ Ollama 仅 Embedding)

模式 3: 纯 API 部署
  最低: 8GB RAM (无 GPU 要求)
  组件: PG + Redis + App
```

### 6.3 新增依赖

```toml
# pyproject.toml 新增
dependencies = [
    # 已有依赖保持不变 ...
    "openai>=1.0.0",           # OpenAI 兼容 API 客户端
    "redis>=5.0.0",            # 缓存
    "structlog>=24.0.0",       # 结构化日志
    "pymupdf>=1.24.0",         # PDF 解析 (比 pdfplumber 快 4x)
    "python-docx>=1.0.0",      # Word 解析
    "tiktoken>=0.7.0",         # Token 计数
]
```

---

## 7. 上下文窗口 Token 预算（完整规则）

### 7.1 静态分配 (28K 窗口基准)

```
┌─────────────────────────────────────────────────────┐
│  段                   │ 默认 Token │ 优先级   │ 可压缩  │
├─────────────────────────────────────────────────────┤
│  System Prompt        │    800     │ required │ 否     │
│  记忆上下文             │  1,500     │ high     │ 是     │
│  检索结果 (法条)        │  8,000     │ highest  │ 分两层  │
│  当前对话历史           │  3,000     │ medium   │ 是     │
│  用户问题              │    500     │ required │ 否     │
│  生成预留空间           │ 12,000     │ required │ 否     │
├─────────────────────────────────────────────────────┤
│  合计                  │ 25,800     │          │       │
│  弹性空间               │  2,200     │          │       │
└─────────────────────────────────────────────────────┘
```

### 7.2 动态调整 (根据查询复杂度)

```
简单查询 (法条查阅):
  检索 → 3K, 记忆 → 2K, 历史 → 4K, 生成预留 → 17K

一般查询 (案例咨询):
  检索 → 5K, 记忆 → 2K, 历史 → 3K, 生成预留 → 16K

复杂查询 (案情分析):
  检索 → 8K, 记忆 → 1K, 历史 → 1K, 生成预留 → 14K

对比分析 (法条对比):
  检索 → 10K, 记忆 → 500, 历史 → 500, 生成预留 → 13K
```

### 7.3 检索结果分层打包

```
层级 1 (必填): 精排 Top-5 法条原文 → 占检索预算 70%
层级 2 (有空间时): 相邻条文扩展 (±2) → 占检索预算 85% 上限
层级 3 (有空间时): 典型案例 Top-2 → 剩余预算

超出上限时从后往前裁剪
```

### 7.4 对话历史压缩

```
当前轮 → 保留原文
近 3 轮 → 保留原文，单条截断到 300 字符
3-6 轮 → LLM 压缩为 1-2 句摘要
6 轮以上 → 仅保留关键实体列表
```

---

## 8. API 设计

### 8.1 新增端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/knowledge/upload` | 上传文档 (multipart/form-data) |
| `GET` | `/api/knowledge/status/{task_id}` | 查询处理状态 |
| `GET` | `/api/knowledge/documents` | 文档列表 (分页) |
| `DELETE` | `/api/knowledge/{doc_id}` | 删除文档及向量 |
| `POST` | `/api/knowledge/reindex` | 重建全量索引 |
| `GET` | `/api/conversations` | 历史会话列表 |
| `GET` | `/api/conversations/{session_id}` | 会话详情 |
| `DELETE` | `/api/conversations/{session_id}` | 删除会话 |
| `POST` | `/api/feedback` | 提交回答反馈 (1-5 星 + 标签) |

### 8.2 Chat 请求扩展

```json
{
  "query": "工伤认定标准是什么",
  "session_id": "uuid",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "top_k": 5,
  "enable_memory": true,
  "enable_faq_cache": true
}
```

### 8.3 Chat 响应扩展

```json
{
  "query": "工伤认定标准是什么",
  "answer": "...",
  "sources": [...],
  "is_casual": false,
  "cache_hit": false,
  "confidence": 0.92,
  "memory_used": 1,
  "memory_summary": "用户之前咨询过工伤赔偿相关问题",
  "request_id": "uuid"
}
```

---

## 9. 验收标准

- [x] 多模型切换：`.env` 修改 `LLM_BACKEND` 即可切换，无需改代码
- [ ] 对话记忆：用户 A 新建对话后，系统能引用 30 分钟前对话 B 的关键信息
- [ ] FAQ 缓存：同一问题第二次查询时，命中缓存直接返回，延迟 < 100ms
- [ ] 文档上传：上传 PDF → 预览 → 确认入库 → 可检索，全流程可用
- [ ] 意图识别：法律查询 / 案例参考 / 超出范围 三类输出正确率 > 90%
- [ ] Token 预算：任何查询的最终 Prompt 不超过模型上下文限制
- [ ] 幻觉防御：检索置信度 < 0.7 时回复"超出知识库范围"而非编造
- [ ] 可观测性：query_logs 表完整记录每次查询的 5 个阶段耗时
- [ ] 不引入回归：174 个已有单元测试全通过
- [ ] 部署验证：`docker compose up -d` 一键启动全套服务

---

## 10. 附录：保留与删除

### 保留
- 所有现有测试用例 (174 个)
- 所有现有文档 (adr-001、adr-002、评测报告等)
- Docker Compose 部署方式
- LangGraph Agent 工作流框架
- JWT 认证体系
- 前端 Vue 3 + Pinia 体系

### 删除/替代
- `src/llm/client.py` → 重构为 `ollama_backend.py` + `openai_backend.py`
- `src/embedding/vector_store.py` (FAISS 管理) → 移至 `src/knowledge/index_manager.py` (pgvector 管理)
- `src/rag/retriever.py` (FAISS 检索器) → 重写为 pgvector 检索器
- `data/vector_store/` (FAISS 索引文件) → 数据迁移到 pgvector
- `scripts/build_index.py` → 替换为 `src/knowledge/ingestion/pipeline.py`

---

## 11. 步骤 1 审查记录 (2026-07-23)

| # | 发现 | 严重 | 修复 |
|---|------|------|------|
| 1 | `LLMAdapter` 不处理 `LLMMessage` 对象，`graph.py`/`engine.py` 传历史会 `AttributeError` | 🔴致命 | `_normalize_history()` 转换 + 6 测试 |
| 2 | `dependencies._create_embedder()` 注释写"回退到 LLM_BACKEND" | 🟢轻微 | 修正为"独立于 LLM_BACKEND" |
| 3 | `LawAgentGraph` 类型标注 `llm: LawLLM` 过时 | 🟡中等 | 运行时无影响，Phase2 重构修正 |

**审查通过项**：
- 无循环导入
- LLM/Embedding 后端独立选型正确
- `.env` 在 `.gitignore` 中
- 旧代码零破坏性变更
