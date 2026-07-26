"""查询改写：将用户口语化表述规范化为法律检索查询。

设计原则（法律系统）：
- 仅做术语规范化，不改变原意，不引入未提及的法律概念。
- 大众不会说法言法语，故改写对召回必要；但改写结果必须由用户确认，
  将"算法黑箱"转为"人机协作"，宁可牺牲部分召回也要保证精度。
- 本模块不写入 LangGraph（因确认需要人机往返），而是作为前端开关控制
  的前置步骤（/api/rewrite）独立调用。
"""
from __future__ import annotations

import logging

from src.agents.prompts import REWRITE_PROMPT

logger = logging.getLogger(__name__)


def rewrite_query(llm, query: str) -> str:
    """用 LLM 将 query 改写为规范法律检索语句。失败则回退原句。"""
    prompt = REWRITE_PROMPT.format(query=query)
    try:
        out = llm.chat(prompt)
    except Exception as e:
        logger.warning("改写失败，回退原句: %s", e)
        return query
    out = (out or "").strip()
    # 去掉模型可能加的引号 / 书名号外壳
    if len(out) >= 2 and out[0] in "\"“'‘" and out[-1] in "\"”'’":
        out = out[1:-1].strip()
    # 去掉模型可能加的前缀（如 "改写："、"回答："）
    for prefix in ("改写：", "改写:", "改写", "回答：", "回答:", "结果：", "结果:"):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
            break
    # 去掉残留换行 / 多余空白
    out = " ".join(out.split())
    return out or query
