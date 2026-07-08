"""
RAG 问答测试 & 交互式演示。

前置条件:
    1. Ollama 已启动，且有 qwen2.5:7b + nomic-embed-text
    2. 已构建 FAISS 索引: uv run python scripts/build_index.py build

用法:
    uv run python scripts/test_rag.py demo        # 交互式问答
    uv run python scripts/test_rag.py test        # 运行预设测试用例
    uv run python scripts/test_rag.py build       # 构建索引 + 测试
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore
from src.llm.client import LawLLM
from src.rag.retriever import FAISSRetriever
from src.rag.engine import RAGEngine, RAGAnswer

VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
INDEX_NAME = "law_index"


# ---------------------------------------------------------------------------
# 预设测试用例
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "治安管理处罚的种类有哪些？",
    "行政拘留合并执行最长多少天？",
    "民法典规定的民事行为能力分为哪几种？",
    "什么情况下可以认定为正当防卫？",
    "用人单位在什么情况下可以单方面解除劳动合同？",
    "商标侵权需要承担什么法律责任？",
]


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

def init_engine() -> RAGEngine:
    """初始化 RAG 引擎（加载已有索引）"""
    print("正在加载 LLM ...")
    llm = LawLLM()

    print("正在加载向量库 ...")
    embedder = LawEmbedder()
    store = VectorStore(embedder=embedder, persist_dir=VECTOR_STORE_DIR, index_name=INDEX_NAME)
    if store.load() is None:
        print("\n错误: 未找到 FAISS 索引，请先运行:")
        print("  uv run python scripts/build_index.py build")
        sys.exit(1)

    retriever = FAISSRetriever(store)
    engine = RAGEngine(retriever=retriever, llm=llm, top_k=5)
    print(f"初始化完成 (索引: {store.doc_count} 条向量)\n")
    return engine


def print_answer(result: RAGAnswer):
    """格式化打印问答结果"""
    print(f"\n{'='*60}")
    print(f"[Q] {result.query}")
    print(f"{'='*60}")
    print(f"\n{result.answer}")
    print(f"\n{'-'*60}")
    print(f"[引用来源]")
    print(result.format_sources())
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

def cmd_test():
    """运行预设测试"""
    engine = init_engine()

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] 处理中...")
        result = engine.ask(query)
        print_answer(result)


def cmd_demo():
    """交互式问答"""
    engine = init_engine()

    print("=" * 60)
    print("  法律 RAG 问答系统 (输入 q 退出)")
    print("  支持: 30 部法律 | 4000+ 条文")
    print("=" * 60)

    history = []
    while True:
        try:
            query = input("\n[Q] 请输入法律问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not query:
            continue
        if query.lower() == "q":
            print("再见!")
            break

        # 检索
        docs = engine.retriever.search(query, top_k=engine.top_k)
        prompt = engine._build_prompt(query, docs)

        print(f"\n检索到 {len(docs)} 条相关条文")
        print("─" * 60)

        # 流式输出
        print("[A] ", end="", flush=True)
        full_answer = ""
        for token in engine.llm.chat_stream(prompt):
            print(token, end="", flush=True)
            full_answer += token

        print(f"\n{'─'*60}")
        result = RAGAnswer(query=query, answer=full_answer, sources=docs)
        print(f"[引用来源]")
        print(result.format_sources())
        print()


def cmd_build_and_test():
    """构建索引 + 运行测试"""
    print("=" * 60)
    print("  步骤 1: 构建 FAISS 索引")
    print("=" * 60)

    from src.chunking.parser import build_all_documents
    from src.chunking.chunker import LawChunker

    print("解析法律文档 ...")
    docs = build_all_documents(PROJECT_ROOT / "LawData")

    print("切分 chunk ...")
    chunker = LawChunker()
    chunks = chunker.chunk_documents(docs)
    print(f"生成 {len(chunks)} 个 chunk")

    print("向量化 + 构建索引 ...")
    embedder = LawEmbedder(batch_size=32)
    store = VectorStore(embedder=embedder, persist_dir=VECTOR_STORE_DIR, index_name=INDEX_NAME)
    store.build_from_documents(chunks, show_progress=True)
    store.save()

    print(f"\n索引构建完成 ({store.doc_count} 条)\n")

    # 直接测试
    cmd_test()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Law RAG 问答测试")
    ap.add_argument("command", nargs="?", default="demo",
                    choices=["demo", "test", "build"],
                    help="demo=交互式 | test=预设测试 | build=构建索引+测试")

    args = ap.parse_args()

    print("Law-RAG-Agent 问答系统")
    print(f"模型: qwen2.5:7b | 检索: FAISS | 法律库: 30部/4145条\n")

    if args.command == "demo":
        cmd_demo()
    elif args.command == "test":
        cmd_test()
    elif args.command == "build":
        cmd_build_and_test()


if __name__ == "__main__":
    main()
