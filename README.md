# Law-RAG-Agent

基于本地大语言模型的法律法规智能问答系统，集成 RAG 检索增强与多 Agent 任务调度。

## 技术栈

- **后端**：Python 3.12+ / FastAPI / LangChain / LangGraph
- **向量检索**：FAISS + bge-large-zh-v1.5
- **LLM**：Ollama (Qwen2.5:7B)
- **元数据**：SQLite
- **容器化**：Docker Compose

## 快速启动

```bash
uv sync
uv run uvicorn api.main:app --reload
```
