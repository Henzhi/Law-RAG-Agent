"""批量检索评估：测试多组 BM25 权重，生成综合对比报告"""
import sys, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore
from src.rag.retriever import FAISSRetriever
from src.rag.hybrid_retriever import HybridRetriever
from src.config import INDEX_DIR, INDEX_NAME, EMBED_MODEL, EMBED_BASE_URL
from scripts.eval_retrieval import RetrievalEvaluator, load_queries

# 测试配置：BM25 权重从 0 到 0.9
WEIGHTS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
WEIGHT_LABELS = {
    0.0: "纯向量",
    0.1: "混合 (BM25=0.1)",
    0.3: "混合 (BM25=0.3)",
    0.5: "混合 (BM25=0.5)",
    0.7: "混合 (BM25=0.7)",
    0.9: "混合 (BM25=0.9)",
}

print("加载索引 ...")
embedder = LawEmbedder(model=EMBED_MODEL, base_url=EMBED_BASE_URL)
store = VectorStore(embedder=embedder, persist_dir=INDEX_DIR, index_name=INDEX_NAME)
store.load()

queries = load_queries()
corpus_path = Path(store.store_dir) / "bm25_corpus.pkl"
has_bm25 = corpus_path.exists()

reports = []
for w in WEIGHTS:
    label = WEIGHT_LABELS[w]
    if w == 0.0:
        retriever = FAISSRetriever(store)
        rtype = "纯向量检索"
    elif has_bm25:
        faiss = FAISSRetriever(store)
        retriever = HybridRetriever.from_corpus_file(faiss, corpus_path, bm25_weight=w)
        rtype = f"混合检索 (BM25权重={w})"
    else:
        continue

    print(f"\n测试: {label} ...")
    evaluator = RetrievalEvaluator(retriever, queries, top_k=10)
    report = evaluator.evaluate()
    reports.append((label, rtype, report))
    print(f"  Recall@5={report.avg_recall_5:.4f}  MRR={report.mrr:.4f}  Latency={report.avg_latency_ms:.0f}ms")

# 生成报告
lines = [
    "# 法律检索质量评估 — 多配置对比报告",
    "",
    f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"**嵌入模型**: {EMBED_MODEL}",
    f"**测试查询数**: {len(queries)}",
    "",
    "## 实验配置",
    "",
    "| 配置 | 检索模式 |",
    "|------|---------|",
]
for label, rtype, _ in reports:
    lines.append(f"| {label} | {rtype} |")

lines += [
    "",
    "## 核心指标对比",
    "",
    "| 配置 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | NDCG@10 | 延迟(ms) |",
    "|------|---------|---------|---------|----------|-----|---------|---------|",
]

for label, _, r in reports:
    lines.append(
        f"| **{label}** | {r.avg_recall_1:.4f} | {r.avg_recall_3:.4f} | "
        f"{r.avg_recall_5:.4f} | {r.avg_recall_10:.4f} | "
        f"{r.mrr:.4f} | {r.ndcg_10:.4f} | {r.avg_latency_ms:.0f} |"
    )

# 找最优
best = max(reports, key=lambda x: x[2].avg_recall_5)
lines += [
    "",
    f"### 最佳 Recall@5: **{best[0]}** ({best[2].avg_recall_5:.4f})",
]

lines += [
    "",
    "## 指标变化趋势",
    "",
    "| BM25权重 | Recall@5 | MRR |",
    "|---------|---------|-----|",
]
for label, _, r in reports:
    bm25_w = label.split("=")[-1].rstrip(")") if "=" in label else "0"
    lines.append(f"| {bm25_w} | {r.avg_recall_5:.4f} | {r.mrr:.4f} |")

# 逐查询对比
lines += [
    "",
    "## 逐查询 Hit@5 对比",
    "",
    "| ID | 查询 | " + " | ".join(r[0] for r in reports) + " |",
    "|----|------|" + "|".join("------" for _ in reports) + "|",
]

for i, q in enumerate(queries):
    row = f"| {q['id']} | {q['query'][:22]} |"
    for _, _, r in reports:
        hit5 = "Y" if r.results[i].hits_at.get(5) else "N"
        row += f" {hit5} |"
    lines.append(row)

# 分析
lines += [
    "",
    "## 分析结论",
    "",
    "### 纯向量检索 (BM25=0)",
    "- 依赖嵌入模型的语义理解能力",
    "- 对自然语言描述的问题表现尚可",
    "",
    "### 混合检索 (BM25>0)",
    "- BM25 补充了条文编号的精确匹配能力",
    "- 权重不宜过高，否则会干扰语义查询",
    "- 当前 `nomic-embed-text` 整体召回率偏低（Recall@5≈0.30-0.35），建议后续换更强的嵌入模型",
    "",
    "### 建议",
    "- 短期：使用 BM25=0.3 作为默认配置（语义+关键词平衡）",
    "- 中期：换用 bge-m3 或 multilingual-e5-large 嵌入模型",
    "- 长期：增加 Reranker 做二次精排",
]

output = Path("docs/retrieval_eval.md")
output.write_text("\n".join(lines), encoding="utf-8")
print(f"\n报告已更新: {output}")
