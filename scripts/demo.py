"""
演示脚本：5 分钟全流程演示 Law-RAG-Agent 核心功能。

用法:
    uv run python scripts/demo.py
    uv run python scripts/demo.py --base http://localhost:8000

需要: 后端已启动 (uvicorn src.api.main:app)
"""
from __future__ import annotations

import json
import sys
import time
import argparse

try:
    import requests
except ImportError:
    print("[FATAL] pip install requests")
    sys.exit(1)


# ============================================================
# 视觉辅助
# ============================================================

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"
SEP = "─" * 62


def title(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*62}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*62}{RESET}\n")


def step(n: int, text: str) -> None:
    print(f"\n{BOLD}{GREEN}>>> Step {n}: {text}{RESET}")
    print(f"{SEP}")


def info(label: str, value: str) -> None:
    print(f"  {YELLOW}{label}:{RESET} {value}")


def truncate(text: str, n: int = 120) -> str:
    return text[:n] + "..." if len(text) > n else text


def wait(seconds: float = 0.5) -> None:
    time.sleep(seconds)


# ============================================================
# 演示步骤
# ============================================================

def run(base_url: str) -> None:
    # ---- 开场 ----
    title("Law-RAG-Agent · 法律法规智能问答系统 · 演示")
    print(f"  后端: {base_url}")
    print(f"  技术栈: FastAPI + LangGraph + FAISS + Qwen2.5:7b + bge-m3")
    wait(1)

    # ---- Step 1: 健康检查 ----
    step(1, "健康检查 — 系统状态")
    try:
        r = requests.get(f"{base_url}/api/health", timeout=10)
        d = r.json()
        info("状态", d.get("status", "?"))
        info("索引就绪", str(d.get("index_ready", "?")))
        info("文档数", str(d.get("doc_count", "?")))
        info("模型", d.get("llm_model", "?"))
    except Exception as e:
        print(f"  [ERROR] {e}")
        return
    wait(1)

    # ---- Step 2: 闲聊路由 ----
    step(2, "闲聊路由 — 不检索，直接回复")
    try:
        r = requests.post(
            f"{base_url}/api/chat",
            json={"query": "你好，你能帮我做什么？", "top_k": 3},
            timeout=180,
        )
        d = r.json()
        is_casual = d.get("is_casual", False)
        info("路由结果", "闲聊模式 (跳过检索)" if is_casual else "法律模式")
        info("回答", truncate(d.get("answer", ""), 150))
    except Exception as e:
        print(f"  [ERROR] {e}")
    wait(1)

    # ---- Step 3: 法律问答 ----
    step(3, "法律问答 — 检索 + 生成")
    query = "治安管理处罚的种类有哪些"
    print(f"  提问: {query}")
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/api/chat",
            json={"query": query, "top_k": 5},
            timeout=180,
        )
        elapsed = time.perf_counter() - t0
        d = r.json()
        info("耗时", f"{elapsed:.0f}s")
        info("回答", truncate(d.get("answer", ""), 200))
        sources = d.get("sources", [])
        info("引用条数", str(len(sources)))
        for i, s in enumerate(sources[:3]):
            info(f"  来源{i+1}", s.get("citation", "?"))
    except Exception as e:
        print(f"  [ERROR] {e}")
    wait(1)

    # ---- Step 4: 法条号直接检索 ----
    step(4, "法条号直接检索 — 精准命中")
    query = "刑法第二十条"
    print(f"  提问: {query}")
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base_url}/api/chat",
            json={"query": query, "top_k": 3},
            timeout=180,
        )
        elapsed = time.perf_counter() - t0
        d = r.json()
        info("耗时", f"{elapsed:.0f}s")
        info("回答", truncate(d.get("answer", ""), 200))
    except Exception as e:
        print(f"  [ERROR] {e}")
    wait(1)

    # ---- Step 5: 多轮对话 ----
    step(5, "多轮对话 — 上下文保持")
    q1 = "劳动合同可以约定试用期多久"
    print(f"  第1轮: {q1}")
    try:
        r1 = requests.post(
            f"{base_url}/api/chat",
            json={"query": q1, "top_k": 3},
            timeout=180,
        )
        a1 = r1.json().get("answer", "")[:120]
        info("第1轮回答", truncate(a1, 120))
    except Exception as e:
        print(f"  [ERROR] {e}")
        a1 = ""

    wait(0.5)
    q2 = "试用期内单位要交社保吗"
    print(f"  第2轮: {q2}")
    try:
        history = [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
        ]
        r2 = requests.post(
            f"{base_url}/api/chat",
            json={"query": q2, "history": history, "top_k": 3},
            timeout=180,
        )
        info("第2轮回答", truncate(r2.json().get("answer", ""), 150))
    except Exception as e:
        print(f"  [ERROR] {e}")
    wait(1)

    # ---- Step 6: 流式问答 ----
    step(6, "流式问答 (SSE) — 逐字输出")
    query = "行政拘留最长多少天"
    print(f"  提问: {query}")
    t0 = time.perf_counter()
    first_token_ms = 0
    got_first = False
    answer_text = ""
    source_count = 0
    try:
        r = requests.post(
            f"{base_url}/api/chat/stream",
            json={"query": query, "top_k": 3},
            timeout=180,
            stream=True,
        )
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
                t = event.get("type", "")
                if t == "meta":
                    source_count = len(event.get("sources", []))
                elif t == "token":
                    if not got_first:
                        got_first = True
                        first_token_ms = (time.perf_counter() - t0) * 1000
                    answer_text += event.get("content", "")
                elif t == "error":
                    info("错误", event.get("content", ""))
            except json.JSONDecodeError:
                pass
        elapsed = time.perf_counter() - t0
        info("首字延迟", f"{first_token_ms:.0f}ms")
        info("总耗时", f"{elapsed:.0f}s")
        info("引用条数", str(source_count))
        info("流式回答", truncate(answer_text, 200))
    except Exception as e:
        print(f"  [ERROR] {e}")
    wait(1)

    # ---- 总结 ----
    title("演示总结")
    print(f"  1. 闲聊路由: 自动跳过检索，直接回复")
    print(f"  2. 法律问答: 检索->Reranker->生成，引用法条+条款号")
    print(f"  3. 法条号检索: 直接输入「刑法第二十条」精准命中")
    print(f"  4. 多轮对话: 追问上下文保持，试用期→社保")
    print(f"  5. 流式输出: SSE 逐字渲染，首字延迟 < 1s")
    print(f"")
    print(f"  知识库: 30部法律 / 3753条文档")
    print(f"  检索质量: Recall@5=80% / MRR=0.65")
    print(f"  回答质量: 综合评分 0.817 / 幻觉率 0%")
    print(f"")
    print(f"  GitHub: https://github.com/Henzhi/Law-RAG-Agent")
    print(f"{CYAN}{'='*62}{RESET}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Law-RAG-Agent 演示脚本")
    ap.add_argument("--base", default="http://localhost:8000", help="后端地址")
    args = ap.parse_args()
    run(args.base)
