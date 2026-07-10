"""
法律文档索引构建脚本。

所有模型配置从 .env / src/config.py 读取。
命令行参数可覆盖 .env 中的设置。

用法:
    uv run scripts/build_index.py build          # 全量构建
    uv run scripts/build_index.py preview        # 预览解析结果
    uv run scripts/build_index.py search         # 检索测试
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    LLM_MODEL, EMBED_MODEL, EMBED_BASE_URL, EMBED_BATCH_SIZE,
    RETRIEVAL_TOP_K, RETRIEVAL_HYBRID_ENABLED, INDEX_NAME, INDEX_DIR,
)
from src.chunking.parser import LawParser, build_all_documents, print_hierarchy
from src.chunking.chunker import LawChunker, ChunkConfig
from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore

LAW_DATA_DIR = PROJECT_ROOT / "LawData"


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def cmd_preview(args: argparse.Namespace) -> None:
    parser = LawParser()
    all_docs = build_all_documents(LAW_DATA_DIR)
    print(f"\n共解析 {len(all_docs)} 部法律\n")

    for doc in all_docs:
        if args.law_name and args.law_name not in doc.title:
            continue
        print_hierarchy(doc)
        print()

    chunker = LawChunker()
    all_chunks = chunker.chunk_documents(all_docs)
    article_chunks = [c for c in all_chunks if c.metadata.get("chunk_type") == "article"]
    summary_chunks = [c for c in all_chunks if c.metadata.get("chunk_type") == "chapter_summary"]
    print(f"\n切分统计:")
    print(f"  法条级 chunk: {len(article_chunks)}")
    print(f"  章级摘要 chunk: {len(summary_chunks)}")
    print(f"  总计: {len(all_chunks)}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("  Law-RAG-Agent 索引构建")
    print(f"  Embedding: {args.embed_model or EMBED_MODEL}")
    print(f"  混合检索: {'开启' if RETRIEVAL_HYBRID_ENABLED else '关闭'}")
    print("=" * 60)

    # 1. 解析
    print("\n[1/5] 解析法律文档 ...")
    parser = LawParser()
    all_docs = build_all_documents(LAW_DATA_DIR)
    print(f"共解析 {len(all_docs)} 部法律，{sum(len(d.articles) for d in all_docs)} 条条文")

    # 2. 切分
    print("\n[2/5] 切分文档 ...")
    chunker = LawChunker(ChunkConfig(
        min_chunk_chars=args.min_chunk,
        max_chunk_chars=args.max_chunk,
        merge_short_articles=not args.no_merge,
        add_chapter_summary=not args.no_summary,
    ))
    all_chunks = chunker.chunk_documents(all_docs)
    print(f"生成 {len(all_chunks)} 个文档片段")

    # 3. 向量化 + FAISS
    print("\n[3/5] 向量化并构建 FAISS 索引 ...")
    embedder = LawEmbedder(
        model=args.embed_model or EMBED_MODEL,
        base_url=args.ollama_url or EMBED_BASE_URL,
        batch_size=args.batch_size or EMBED_BATCH_SIZE,
    )

    store = VectorStore(
        embedder=embedder,
        persist_dir=INDEX_DIR,
        index_name=args.index_name or INDEX_NAME,
    )
    store.build_from_documents(all_chunks, show_progress=not args.quiet)
    store.save()

    # 4. BM25 语料（混合检索用）
    if RETRIEVAL_HYBRID_ENABLED:
        print("\n[4/6] 构建 BM25 关键词索引 ...")
        from src.rag.hybrid_retriever import HybridRetriever
        from src.rag.retriever import FAISSRetriever

        texts = [doc.page_content for doc in all_chunks]
        faiss = FAISSRetriever(store)
        ret_docs = [FAISSRetriever._to_retrieved(doc, 0.0) for doc in all_chunks]
        hr = HybridRetriever(
            vector_retriever=faiss,
            corpus_texts=texts,
            corpus_docs=ret_docs,
        )
        corpus_path = Path(store.store_dir) / "bm25_corpus.pkl"
        hr.save_corpus(corpus_path)
        print(f"BM25 语料已保存: {corpus_path}")
    else:
        print("\n[4/6] 跳过 (混合检索未开启)")

    # 5. 保存条文映射（连续片段检索用）
    print("\n[5/6] 保存条文映射 ...")
    from src.rag.adjacent_expander import AdjacentExpander
    article_map = AdjacentExpander.build_article_map(all_docs)
    map_path = Path(store.store_dir) / "article_map.json"
    AdjacentExpander.save_article_map(article_map, map_path)

    # 6. 保存法律分类索引
    print("\n[6/6] 保存法律分类索引 ...")
    law_index = _build_law_index(all_docs)
    index_path = Path(store.store_dir) / "law_index.json"
    import json as _json
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        _json.dump(law_index, f, ensure_ascii=False, indent=2)
    print(f"  法律分类索引已保存: {index_path} ({len(law_index)} 部法律)")

    print("\n" + "=" * 60)
    print(f"  索引名称: {args.index_name or INDEX_NAME}")
    print(f"  文档片段: {store.doc_count}")
    print(f"  嵌入模型: {args.embed_model or EMBED_MODEL}")
    print(f"  检索模式: {'混合 (BM25+向量)' if RETRIEVAL_HYBRID_ENABLED else '纯向量'}")
    print(f"  存储路径: {store.store_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    embedder = LawEmbedder(
        model=args.embed_model or EMBED_MODEL,
        base_url=args.ollama_url or EMBED_BASE_URL,
    )
    store = VectorStore(
        embedder=embedder,
        persist_dir=INDEX_DIR,
        index_name=args.index_name or INDEX_NAME,
    )

    if store.load() is None:
        print("错误: 向量库不存在，请先运行 build 命令")
        sys.exit(1)

    # 构建检索器
    from src.rag.retriever import FAISSRetriever
    corpus_path = Path(store.store_dir) / "bm25_corpus.pkl"
    if RETRIEVAL_HYBRID_ENABLED and corpus_path.exists():
        from src.rag.hybrid_retriever import HybridRetriever
        import pickle
        with open(corpus_path, "rb") as f:
            data = pickle.load(f)
        faiss = FAISSRetriever(store)
        ret_docs = [
            FAISSRetriever._to_retrieved(
                type("Doc", (), {"page_content": t, "metadata": {}})(), 0.0
            )
            for t in data["texts"]
        ]
        retriever = HybridRetriever.from_corpus_file(faiss, ret_docs, corpus_path)
        print(f"检索模式: 混合 (BM25 + 向量)")
    else:
        retriever = FAISSRetriever(store)
        print(f"检索模式: 纯向量")

    while True:
        try:
            query = input("\n请输入法律问题 (q 退出): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not query or query.lower() == "q":
            break

        results = retriever.search(query, k=args.top_k or RETRIEVAL_TOP_K)
        print(f"\n检索结果 (Top {len(results)}):")
        print("-" * 60)
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] 分数: {doc.score:.4f} | {doc.citation}")
            print(f"    内容: {doc.content[:120]}...")


# ---------------------------------------------------------------------------
# 法律分类自动识别
# ---------------------------------------------------------------------------

_CATEGORY_RULES = [
    ("刑法", ["刑法", "刑事", "罪名", "量刑", "犯罪", "刑罚"]),
    ("民法", ["民法", "合同", "物权", "侵权", "婚姻", "继承", "担保", "人格权", "民法典"]),
    ("行政法", ["行政", "治安", "交通", "许可", "处罚", "复议", "国家赔偿", "公务员"]),
    ("经济法", ["经济", "商业", "公司", "反垄断", "证券", "银行", "保险", "票据", "企业", "税收", "税务", "消费者", "产品"]),
    ("社会法", ["劳动", "就业", "社保", "社会", "残疾人", "未成年", "妇女", "老年人"]),
    ("环境资源法", ["环境", "资源", "生态", "水土", "矿产", "森林", "草原", "海洋", "能源"]),
    ("宪法与组织法", ["宪法", "立法", "组织", "选举", "代表", "国旗", "国徽", "民族区域"]),
    ("诉讼法", ["诉讼", "仲裁", "调解", "执行", "复议"]),
]

def _categorize_law(title: str) -> str:
    """根据法律名称自动归类"""
    for category, keywords in _CATEGORY_RULES:
        for kw in keywords:
            if kw in title:
                return category
    return "其他"


def _build_law_index(all_docs: list) -> list[dict]:
    """生成法律分类索引"""
    return [
        {
            "law_name": doc.title,
            "category": _categorize_law(doc.title),
            "article_count": len(doc.articles),
        }
        for doc in all_docs
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Law-RAG-Agent 索引构建工具")
    sub = ap.add_subparsers(dest="command", help="子命令")

    # build
    p_build = sub.add_parser("build", help="全量构建")
    p_build.add_argument("--min-chunk", type=int, default=50)
    p_build.add_argument("--max-chunk", type=int, default=1500)
    p_build.add_argument("--no-merge", action="store_true")
    p_build.add_argument("--no-summary", action="store_true")
    p_build.add_argument("--embed-model", default=None, help=f"默认: {EMBED_MODEL}")
    p_build.add_argument("--ollama-url", default=None, help=f"默认: {EMBED_BASE_URL}")
    p_build.add_argument("--batch-size", type=int, default=None, help=f"默认: {EMBED_BATCH_SIZE}")
    p_build.add_argument("--index-name", default=None, help=f"默认: {INDEX_NAME}")
    p_build.add_argument("--quiet", action="store_true")

    # preview
    p_preview = sub.add_parser("preview", help="预览解析")
    p_preview.add_argument("--law-name", type=str, default="")

    # search
    p_search = sub.add_parser("search", help="检索测试")
    p_search.add_argument("--embed-model", default=None)
    p_search.add_argument("--ollama-url", default=None)
    p_search.add_argument("--index-name", default=None)
    p_search.add_argument("--top-k", type=int, default=None)

    args = ap.parse_args()

    if args.command == "preview":
        cmd_preview(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        # 默认 build
        cmd_build(args)


if __name__ == "__main__":
    main()
