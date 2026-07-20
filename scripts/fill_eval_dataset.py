"""
自动填充测试数据集：逐条运行 RAG 流水线，写入 retrieved_text 和 model_output。

用法:
    uv run python scripts/fill_eval_dataset.py              # 全量填充
    uv run python scripts/fill_eval_dataset.py --limit 5     # 只跑前5条
    uv run python scripts/fill_eval_dataset.py --resume      # 跳过已填充的继续

填充对象: data/eval_dataset.json (131 条)
"""
from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBED_MODEL, EMBED_BASE_URL, EMBED_BATCH_SIZE,
    INDEX_DIR, INDEX_NAME,
    RETRIEVAL_TOP_K,
    LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS,
    RERANK_ENABLED, RERANK_MODEL, RERANK_RECALL_K, RERANK_TOP_K,
)
from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore
from src.rag.retriever import FAISSRetriever, RetrievedDoc
from src.llm.client import LawLLM, LLMConfig

DATASET_PATH = PROJECT_ROOT / "data" / "eval_dataset.json"


def load_retriever():
    """加载 FAISS 检索器 + 可选 Reranker"""
    print(f"[1/3] 加载 embedding 模型: {EMBED_MODEL} ...")
    embedder = LawEmbedder(
        model=EMBED_MODEL, base_url=EMBED_BASE_URL, batch_size=EMBED_BATCH_SIZE,
    )
    print(f"[2/3] 加载向量库: {INDEX_DIR / INDEX_NAME} ...")
    store = VectorStore(embedder=embedder, persist_dir=INDEX_DIR, index_name=INDEX_NAME)
    store.load()
    print(f"      文档数: {store.doc_count}")

    retriever = FAISSRetriever(store)

    # 尝试启用 Reranker（需要模型文件已下载）
    if RERANK_ENABLED:
        try:
            from src.rag.reranker import Reranker, RerankRetriever
            reranker = Reranker(model_name=RERANK_MODEL)
            retriever = RerankRetriever(
                base_retriever=retriever,
                reranker=reranker,
                recall_k=RERANK_RECALL_K,
                top_k=RERANK_TOP_K,
            )
            print(f"      已启用 Reranker: {RERANK_MODEL}")
        except Exception as e:
            print(f"      [WARN] Reranker 不可用: {e}")
            print(f"      将使用 chunk_type 过滤替代（效果相当）")

    return retriever


def load_llm():
    """加载 LLM 客户端"""
    print(f"[3/3] 加载 LLM: {LLM_MODEL} ...")
    config = LLMConfig(
        temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P, num_predict=LLM_MAX_TOKENS,
    )
    return LawLLM(model=LLM_MODEL, base_url=LLM_BASE_URL, config=config)


def run_one(query: str, retriever, llm: LawLLM) -> tuple[str, str]:
    """对单条 query 运行检索 + 生成，返回 (retrieved_text, model_output)

    过滤策略:
    1. 跳过 chunk_type == 'chapter_summary' 的章级摘要（噪声大）
    2. 限制单条 chunk 内容 ≤ 1500 字符（避免大块吞掉 prompt）
    """
    # 检索（多召回一些候选，后续过滤）
    docs = retriever.search(query, top_k=max(RETRIEVAL_TOP_K * 2, 10))

    # 过滤章级摘要 + 截断过长内容
    filtered: list[RetrievedDoc] = []
    for doc in docs:
        # 跳过章级摘要 chunk（含几十条不相关条文）
        if getattr(doc, "chunk_type", "") == "chapter_summary":
            continue
        # 截断过长内容
        if len(doc.content) > 1500:
            doc.content = doc.content[:1500] + "\n...(内容过长已截断)"
        filtered.append(doc)
        if len(filtered) >= RETRIEVAL_TOP_K:
            break

    # 不够 top_k 时用原始结果补齐
    if len(filtered) < RETRIEVAL_TOP_K:
        for doc in docs:
            if doc not in filtered:
                if len(doc.content) > 1500:
                    doc.content = doc.content[:1500] + "\n...(内容过长已截断)"
                filtered.append(doc)
            if len(filtered) >= RETRIEVAL_TOP_K:
                break

    # 拼接检索文本
    chunks = []
    for i, doc in enumerate(filtered):
        chunks.append(f"[{i+1}] {doc.citation}\n{doc.content}")
    retrieved_text = "\n\n---\n\n".join(chunks)

    # 构建 prompt 并生成
    context = "\n\n".join(
        f"条文{i+1} ({doc.citation}):\n{doc.content}"
        for i, doc in enumerate(filtered)
    )
    prompt = f"""你是一位专业的中国法律助手。请根据以下法律条文，准确回答用户的问题。

## 要求
1. 引用法律名称和条款号
2. 基于条文内容，不编造
3. 回答简洁清晰

## 相关法律条文
{context}

## 用户问题
{query}

请回答："""

    answer = llm.chat(prompt, system_prompt="你是一位专业的中国法律助手，请根据提供的条文准确回答。")
    return retrieved_text, answer.strip()


def main():
    ap = argparse.ArgumentParser(description="自动填充测试数据集")
    ap.add_argument("--limit", type=int, default=0, help="只填充前 N 条 (0=全部)")
    ap.add_argument("--resume", action="store_true", help="跳过已填充的条目")
    ap.add_argument("--start", type=int, default=0, help="从第 N 条开始 (0-based)")
    args = ap.parse_args()

    # 加载数据
    print(f"[0/4] 加载数据集: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    total = len(dataset)
    print(f"      共 {total} 条")

    # 确定范围
    if args.resume:
        skipped = sum(1 for d in dataset if d.get("model_output"))
        print(f"      已填充 {skipped} 条，将跳过")
    start = args.start
    end = start + args.limit if args.limit > 0 else total

    # 加载模型
    retriever = load_retriever()
    llm = load_llm()
    print(f"      开始填充 (第 {start+1}~{min(end,total)} 条)\n")

    done = 0
    errors = 0
    t_total_start = time.perf_counter()

    for i in range(start, min(end, total)):
        item = dataset[i]

        # 跳过已填充
        if args.resume and item.get("model_output"):
            continue

        t0 = time.perf_counter()
        q = item["question"]
        label = f"[{i+1}/{total}]"

        try:
            retrieved_text, model_output = run_one(q, retriever, llm)
            elapsed = (time.perf_counter() - t0)

            item["retrieved_text"] = retrieved_text
            item["model_output"] = model_output
            done += 1

            # 进度显示
            ans_preview = model_output[:60].replace("\n", " ")
            print(f"  {label} OK  {elapsed:.0f}s | {ans_preview}...")

        except Exception as e:
            errors += 1
            item["retrieved_text"] = f"ERROR: {e}"
            item["model_output"] = f"ERROR: {e}"
            elapsed = (time.perf_counter() - t0)
            print(f"  {label} ERR {elapsed:.0f}s | {type(e).__name__}: {e}")

        # 每 5 条保存一次（防止中断丢失）
        if (done + errors) % 5 == 0:
            with open(DATASET_PATH, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            avg = (time.perf_counter() - t_total_start) / (done + errors)
            remaining = (end - i - 1) * avg
            print(f"  [已保存] {done} 完成 / {errors} 失败 | 预计剩余 {remaining/60:.0f}min")

    # 最终保存
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    total_elapsed = time.perf_counter() - t_total_start
    print(f"\n{'='*50}")
    print(f"完成: {done} 条 / 失败: {errors} 条 / 总耗时: {total_elapsed/60:.1f}min")
    print(f"文件已更新: {DATASET_PATH}")


if __name__ == "__main__":
    main()
