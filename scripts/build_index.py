"""
法律文档索引构建脚本。

用法:
    uv run scripts/build_index.py                    # 全量构建
    uv run scripts/build_index.py --preview          # 预览解析结果（不构建索引）
    uv run scripts/build_index.py --law-name 刑法    # 只查看特定法律的结构
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.parser import LawParser, build_all_documents, print_hierarchy
from src.chunking.chunker import LawChunker, ChunkConfig
from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore

# 默认路径
LAW_DATA_DIR = PROJECT_ROOT / 'LawData'
VECTOR_STORE_DIR = PROJECT_ROOT / 'data' / 'vector_store'


def cmd_preview(args: argparse.Namespace) -> None:
    """预览模式：解析 + 显示结构，不构建索引"""
    parser = LawParser()
    all_docs = build_all_documents(LAW_DATA_DIR)
    print(f'\n共解析 {len(all_docs)} 部法律\n')

    for doc in all_docs:
        if args.law_name and args.law_name not in doc.title:
            continue
        print_hierarchy(doc)
        print()

    # 显示切分统计
    chunker = LawChunker()
    all_chunks = chunker.chunk_documents(all_docs)
    article_chunks = [c for c in all_chunks if c.metadata.get('chunk_type') == 'article']
    summary_chunks = [c for c in all_chunks if c.metadata.get('chunk_type') == 'chapter_summary']
    print(f'\n切分统计:')
    print(f'  法条级 chunk: {len(article_chunks)}')
    print(f'  章级摘要 chunk: {len(summary_chunks)}')
    print(f'  总计: {len(all_chunks)}')

    # 显示几个示例 chunk
    if all_chunks:
        print(f'\n--- 示例 chunk (前 3 条) ---')
        for i, chunk in enumerate(all_chunks[:3]):
            print(f'\n[Chunk {i+1}]')
            print(f'  元数据: {chunk.metadata}')
            print(f'  内容: {chunk.page_content[:200]}...')


def cmd_build(args: argparse.Namespace) -> None:
    """全量构建：解析 → 切分 → 向量化 → 保存"""
    print('=' * 60)
    print('  Law-RAG-Agent 索引构建')
    print('=' * 60)

    # 1. 解析文档
    print('\n[1/4] 解析法律文档...')
    parser = LawParser()
    all_docs = build_all_documents(LAW_DATA_DIR)
    print(f'共解析 {len(all_docs)} 部法律')

    total_articles = sum(len(d.articles) for d in all_docs)
    print(f'总计 {total_articles} 条法律条文')

    # 2. 切分
    print('\n[2/4] 切分文档...')
    chunker = LawChunker(ChunkConfig(
        min_chunk_chars=args.min_chunk,
        max_chunk_chars=args.max_chunk,
        merge_short_articles=not args.no_merge,
        add_chapter_summary=not args.no_summary,
    ))
    all_chunks = chunker.chunk_documents(all_docs)
    print(f'生成 {len(all_chunks)} 个文档片段')

    # 3. 向量化 + 构建 FAISS
    print('\n[3/4] 向量化并构建 FAISS 索引...')
    embedder = LawEmbedder(
        model=args.embed_model,
        base_url=args.ollama_url,
        batch_size=args.batch_size,
    )

    store = VectorStore(
        embedder=embedder,
        persist_dir=VECTOR_STORE_DIR,
        index_name=args.index_name,
    )

    store.build_from_documents(all_chunks, show_progress=not args.quiet)

    # 4. 持久化
    print('\n[4/4] 保存向量库...')
    store.save()

    print('\n' + '=' * 60)
    print(f'  构建完成!')
    print(f'  索引名称: {args.index_name}')
    print(f'  文档片段数: {store.doc_count}')
    print(f'  存储路径: {store.store_dir}')
    print('=' * 60)


def cmd_search(args: argparse.Namespace) -> None:
    """检索测试"""
    embedder = LawEmbedder(
        model=args.embed_model,
        base_url=args.ollama_url,
    )
    store = VectorStore(
        embedder=embedder,
        persist_dir=VECTOR_STORE_DIR,
        index_name=args.index_name,
    )

    if store.load() is None:
        print('错误: 向量库不存在，请先运行 build 命令构建索引')
        sys.exit(1)

    while True:
        try:
            query = input('\n请输入法律问题 (输入 q 退出): ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n再见!')
            break

        if not query or query.lower() == 'q':
            break

        results = store.search_with_score(query, k=args.top_k)

        print(f'\n检索结果 (Top {len(results)}):')
        print('-' * 60)
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            print(f'\n[{i}] 相似度: {score:.4f}')
            print(f'    法律: {meta.get("law_name", "")}')
            print(f'    章节: {meta.get("chapter", "")}')
            print(f'    条文: {meta.get("article_range", "")}')
            print(f'    内容: {doc.page_content[:150]}...')
            if i >= args.top_k:
                break


def main():
    ap = argparse.ArgumentParser(description='Law-RAG-Agent 法律文档索引构建工具')
    sub = ap.add_subparsers(dest='command', help='子命令')

    # ---- build ----
    p_build = sub.add_parser('build', help='全量构建向量索引')
    p_build.add_argument('--min-chunk', type=int, default=50,
                         help='最小 chunk 长度（字符）')
    p_build.add_argument('--max-chunk', type=int, default=1500,
                         help='最大 chunk 长度（字符）')
    p_build.add_argument('--no-merge', action='store_true',
                         help='不合并短条文')
    p_build.add_argument('--no-summary', action='store_true',
                         help='不生成章级摘要 chunk')
    p_build.add_argument('--embed-model', default='nomic-embed-text',
                         help='Ollama embedding 模型名称')
    p_build.add_argument('--ollama-url', default='http://localhost:11434',
                         help='Ollama 服务地址')
    p_build.add_argument('--batch-size', type=int, default=32,
                         help='Embedding 批大小')
    p_build.add_argument('--index-name', default='law_index',
                         help='索引名称')
    p_build.add_argument('--quiet', action='store_true',
                         help='安静模式')

    # ---- preview ----
    p_preview = sub.add_parser('preview', help='预览解析结果（不构建索引）')
    p_preview.add_argument('--law-name', type=str, default='',
                           help='只查看包含该关键词的法律')

    # ---- search ----
    p_search = sub.add_parser('search', help='检索测试')
    p_search.add_argument('--embed-model', default='nomic-embed-text')
    p_search.add_argument('--ollama-url', default='http://localhost:11434')
    p_search.add_argument('--index-name', default='law_index')
    p_search.add_argument('--top-k', type=int, default=5)

    args = ap.parse_args()

    if args.command == 'preview':
        cmd_preview(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'build':
        cmd_build(args)
    else:
        # 默认：build
        class DefaultArgs:
            pass
        default = DefaultArgs()
        default.min_chunk = 50
        default.max_chunk = 1500
        default.no_merge = False
        default.no_summary = False
        default.embed_model = 'nomic-embed-text'
        default.ollama_url = 'http://localhost:11434'
        default.batch_size = 32
        default.index_name = 'law_index'
        default.quiet = False
        cmd_build(default)


if __name__ == '__main__':
    main()
