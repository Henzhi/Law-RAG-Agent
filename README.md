# Law-RAG-Agent

基于本地大语言模型的法律法规智能问答系统，集成 RAG 检索增强与多 Agent 任务调度。

## 技术栈

| 层次 | 技术 |
|------|------|
| 后端框架 | Python 3.12+ / FastAPI / LangChain |
| 向量检索 | FAISS + Ollama nomic-embed-text |
| LLM | Ollama (Qwen2.5:7B) |
| 依赖管理 | uv |

## 项目结构

```
Law-RAG-Agent/
├── src/
│   ├── chunking/           # 法律文档解析 + 法条级切分
│   │   ├── parser.py       # 文档结构解析 (编→章→节→条)
│   │   └── chunker.py      # 智能切分 + 章级摘要
│   ├── embedding/          # 向量化
│   │   ├── embedder.py     # Ollama embedding 客户端
│   │   └── vector_store.py # FAISS 向量库
│   ├── llm/                # LLM 客户端
│   │   └── client.py       # Ollama Qwen2.5:7B 封装
│   ├── rag/                # RAG 问答引擎
│   │   ├── retriever.py    # 检索器抽象 (FAISS / pgvector)
│   │   └── engine.py       # 问答管线
│   └── api/                # FastAPI 接口层
│       ├── main.py         # 应用入口
│       ├── routes.py       # API 路由
│       ├── models.py       # 请求/响应模型
│       └── dependencies.py # 依赖注入
├── scripts/
│   ├── build_index.py      # 构建 FAISS 索引
│   ├── test_llm.py         # LLM 测试
│   └── test_rag.py         # RAG 测试 / 交互 demo
├── static/
│   └── index.html          # Web 前端
├── LawData/                # 30 部中国法律原始文本 (4145 条)
└── pyproject.toml
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
uv sync

# 安装 Ollama 并拉取模型
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 2. 构建向量索引

```bash
uv run python scripts/build_index.py build
```

### 3. 启动服务

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

打开浏览器访问 `http://localhost:8000` 即可使用。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/chat` | 法律问答 (返回完整答案 + 引用) |
| `POST` | `/api/chat/stream` | 流式问答 (SSE) |

### 请求示例

```bash
# 单次问答
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "治安管理处罚有哪几种", "top_k": 5}'

# 流式问答
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "行政拘留最长多久", "top_k": 3}'
```

### 响应格式

```json
{
  "query": "治安管理处罚有哪几种",
  "answer": "根据《中华人民共和国治安管理处罚法》...",
  "sources": [
    {
      "law_name": "中华人民共和国治安管理处罚法(2025修订)",
      "chapter": "第二章 处罚的种类和适用",
      "article_range": "第十条",
      "citation": "治安管理处罚法 · 第十条",
      "score": 0.85
    }
  ]
}
```

## 命令行工具

```bash
# 预览解析结果
uv run python scripts/build_index.py preview --law-name 刑法

# LLM 测试
uv run python scripts/test_llm.py --stream

# RAG 交互式问答
uv run python scripts/test_rag.py demo

# RAG 预设测试
uv run python scripts/test_rag.py test
```

## 切分策略

- 以「条」为最小切分单元，保持法律语义完整
- 短于 50 字的连续条文自动合并
- 为每章额外生成摘要 chunk，支持粗粒度检索
- 每个 chunk 携带完整层次元数据 (法律名 → 编 → 章 → 节 → 条文范围)

## 数据

`LawData/` 目录包含 30 部中国法律原文，共 4145 条条文，涵盖民法、刑法、行政法、经济法等多个领域。
