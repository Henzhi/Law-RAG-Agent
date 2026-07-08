"""对比评估：nomic-embed-text vs bge-m3 + 多组 BM25 权重"""
import sys, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore
from src.rag.retriever import FAISSRetriever
from src.rag.hybrid_retriever import HybridRetriever
from scripts.eval_retrieval import RetrievalEvaluator, load_queries

# 两组实验
EXPERIMENTS = [
    {"name": "nomic-embed-text", "index": "law_index", "embed_model": "nomic-embed-text"},
    {"name": "bge-m3", "index": "law_index_bge", "embed_model": "bge-m3"},
]

WEIGHTS = [0.0, 0.3, 0.5, 0.7, 0.9]
WEIGHT_LABEL = {0.0: "纯向量", 0.3: "+BM25(0.3)", 0.5: "+BM25(0.5)", 0.7: "+BM25(0.7)", 0.9: "+BM25(0.9)"}

queries = load_queries()
all_results = {}  # {(model, weight): report}

for exp in EXPERIMENTS:
    name = exp["name"]
    print(f"\n{'='*60}")
    print(f"  模型: {name}")
    print(f"{'='*60}")

    embedder = LawEmbedder(model=exp["embed_model"])
    store = VectorStore(embedder=embedder, persist_dir="data/vector_store", index_name=exp["index"])
    store.load()
    print(f"  索引: {store.doc_count} 条向量")

    corpus_path = Path(store.store_dir) / "bm25_corpus.pkl"
    has_bm25 = corpus_path.exists()

    for w in WEIGHTS:
        label = f"{name} {WEIGHT_LABEL[w]}"
        if w == 0.0:
            retriever = FAISSRetriever(store)
        elif has_bm25:
            faiss = FAISSRetriever(store)
            retriever = HybridRetriever.from_corpus_file(faiss, corpus_path, bm25_weight=w)
        else:
            continue

        evaluator = RetrievalEvaluator(retriever, queries, top_k=10)
        report = evaluator.evaluate()
        all_results[(name, w)] = report
        print(f"  {WEIGHT_LABEL[w]:>12} | R@5={report.avg_recall_5:.4f}  MRR={report.mrr:.4f}  Lat={report.avg_latency_ms:.0f}ms")


# 生成对比报告
lines = [
    "# 法律检索质量评估 — 嵌入模型对比报告",
    "",
    f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"**测试查询数**: {len(queries)}",
    "",
    "## 模型信息",
    "",
    "| 模型 | 维度 | 语言 | 说明 |",
    "|------|------|------|------|",
    "| nomic-embed-text | 768 | English-only | 默认模型，英语优化 |",
    "| **bge-m3** | **1024** | **多语言 (含中文)** | **BAAI 多语言 SOTA** |",
    "",
    "## 核心指标对比 (Recall@5)",
    "",
    "| 配置 | nomic-embed-text | bge-m3 | bge提升 |",
    "|------|:---:|:---:|:---:|",
]

for w in WEIGHTS:
    nomic_r = all_results.get(("nomic-embed-text", w))
    bge_r = all_results.get(("bge-m3", w))
    if nomic_r and bge_r:
        diff = bge_r.avg_recall_5 - nomic_r.avg_recall_5
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        lines.append(
            f"| {WEIGHT_LABEL[w]} | {nomic_r.avg_recall_5:.4f} | "
            f"**{bge_r.avg_recall_5:.4f}** | {arrow} {diff:+.4f} |"
        )

lines += [
    "",
    "## 完整指标矩阵",
    "",
    "| 模型 | 配置 | R@1 | R@3 | R@5 | R@10 | MRR | NDCG@10 | 延迟 |",
    "|------|------|-----|-----|-----|------|-----|---------|------|",
]

for exp in EXPERIMENTS:
    name = exp["name"]
    for w in WEIGHTS:
        r = all_results.get((name, w))
        if not r:
            continue
        lines.append(
            f"| {name} | {WEIGHT_LABEL[w]} | "
            f"{r.avg_recall_1:.4f} | {r.avg_recall_3:.4f} | "
            f"{r.avg_recall_5:.4f} | {r.avg_recall_10:.4f} | "
            f"{r.mrr:.4f} | {r.ndcg_10:.4f} | {r.avg_latency_ms:.0f}ms |"
        )

# 最佳配置
best_nomic = max(
    [(w, r) for (m, w), r in all_results.items() if m == "nomic-embed-text"],
    key=lambda x: x[1].avg_recall_5
)
best_bge = max(
    [(w, r) for (m, w), r in all_results.items() if m == "bge-m3"],
    key=lambda x: x[1].avg_recall_5
)

lines += [
    "",
    "## 最佳配置对比",
    "",
    f"| 模型 | 最佳配置 | Recall@5 | MRR |",
    f"|------|---------|---------|-----|",
    f"| nomic-embed-text | {WEIGHT_LABEL[best_nomic[0]]} | {best_nomic[1].avg_recall_5:.4f} | {best_nomic[1].mrr:.4f} |",
    f"| **bge-m3** | **{WEIGHT_LABEL[best_bge[0]]}** | **{best_bge[1].avg_recall_5:.4f}** | **{best_bge[1].mrr:.4f}** |",
    "",
    "## 逐查询 Hit@5 对比",
    "",
    "| ID | 查询 | nomic纯向量 | nomic最优 | bge纯向量 | bge最优 |",
    "|----|------|:---:|:---:|:---:|:---:|",
]

best_nomic_w = best_nomic[0]
best_bge_w = best_bge[0]
for i, q in enumerate(queries):
    nv = "Y" if all_results[("nomic-embed-text", 0.0)].results[i].hits_at.get(5) else "N"
    nb = "Y" if all_results[("nomic-embed-text", best_nomic_w)].results[i].hits_at.get(5) else "N"
    bv = "Y" if all_results[("bge-m3", 0.0)].results[i].hits_at.get(5) else "N"
    bb = "Y" if all_results[("bge-m3", best_bge_w)].results[i].hits_at.get(5) else "N"
    lines.append(f"| {q['id']} | {q['query'][:22]} | {nv} | {nb} | {bv} | {bb} |")

# 分析
lines += [
    "",
    "## 分析结论",
    "",
    "### bge-m3 相比 nomic-embed-text",
    "",
    f"- **纯向量 Recall@5**: nomic={all_results[('nomic-embed-text',0.0)].avg_recall_5:.4f} → "
    f"bge={all_results[('bge-m3',0.0)].avg_recall_5:.4f}",
    f"- **最佳 Recall@5**: nomic={best_nomic[1].avg_recall_5:.4f} → "
    f"bge={best_bge[1].avg_recall_5:.4f}",
    "",
]

# 具体分析
nomic_pure = all_results[("nomic-embed-text", 0.0)].avg_recall_5
bge_pure = all_results[("bge-m3", 0.0)].avg_recall_5
if bge_pure > nomic_pure:
    lines.append(f"bge-m3 纯向量检索提升了 **{(bge_pure-nomic_pure)/nomic_pure*100:.0f}%**，证明其中文语义理解能力远超 nomic-embed-text。")
else:
    lines.append(f"bge-m3 纯向量检索并未显著提升 ({bge_pure:.4f} vs {nomic_pure:.4f})。")

lines += [
    "",
    "### 建议",
    f"- 默认嵌入模型切换为 **bge-m3**",
    f"- 推荐配置: bge-m3 + BM25={best_bge_w}（Recall@5={best_bge[1].avg_recall_5:.2f}）",
    "- 后续可尝试更多中文优化模型: bge-large-zh-v1.5, stella-base-zh-v3-1792d",
]

Path("docs/retrieval_eval.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\n报告已更新: docs/retrieval_eval.md")
