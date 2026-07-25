"""
统一意图识别模块。

合并原 src/rag/engine.py（正则 + LLM 自省路由）与 src/agents/graph.py（关键词集合）
两份重复实现，提供单一事实来源：

- is_casual_query(query)  -> 是否为明显闲聊/问候（正则判定，供 RAGEngine 路由与元数据标记）
- classify_intent(query)  -> 是否为法律问题（关键词判定，供 LangGraph Agent 路由）
- needs_retrieval(query, llm) -> 是否需要检索（正则 + 长查询快路径 + LLM 自省兜底）
- sanitize_input(query)   -> 输入安全过滤（Prompt 注入防御 + 内容审核）

注意：is_casual_query 与 classify_intent 使用不同算法（历史行为需分别保留，
否则会破坏既有单测），但共用本模块的词典常量，消除之前分散在两处的重复代码。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 词典常量（单一来源）
# ---------------------------------------------------------------------------

# 闲聊类正则模式 — 匹配后直接走 LLM 回复，跳过检索
_CASUAL_PATTERNS = [
    # 问候
    r'^(你好|您好|hi|hello|嗨|早上好|下午好|晚上好|大家好)',
    r'^(在吗|在不|在不在)$',
    # 感谢
    r'^(谢谢|感谢|多谢|thanks|thank)',
    # 告别
    r'^(再见|拜拜|bye|晚安|回头见)',
    # 自我介绍
    r'^(你是谁|你叫什么|你是什么|你的名字|介绍.*自己)',
    r'^(你能做什么|你会什么|你有什么功能|你能干什么)',
    # 纯闲聊
    r'^(今天天气|天气怎么样|讲个笑话|说个笑话)',
    r'^(嗯|哦|好吧|好的|OK|ok)$',
]

# 闲聊短语 — 精确匹配即跳过检索
_CASUAL_PHRASES = {
    # 问候
    "你好", "您好", "hi", "hello", "嗨", "哈喽",
    "早上好", "下午好", "晚上好", "中午好", "晚安", "早",
    "你是谁", "你叫什么", "你是什么", "你的名字",
    "介绍自己", "自我介绍", "你是啥",
    "在吗", "在不在", "在线吗",
    "你能做什么", "你会什么", "你有什么功能",
    "开始", "开始咨询", "测试", "test", "试试",
    # 感谢
    "谢谢", "感谢", "多谢", "thanks", "thank you",
    "非常感谢", "十分感谢", "万分感谢",
    # 告别
    "再见", "拜拜", "bye", "goodbye", "走了", "告辞",
}

# 法律关键词 — 包含任一即走检索
_LEGAL_KEYWORDS = [
    # 法律概念
    "法律", "法条", "法规", "条文", "条款", "规定（法律",
    # 处罚
    "处罚", "罚款", "拘留", "判刑", "刑期", "有期徒刑",
    "无期徒刑", "死刑", "拘役", "管制", "没收", "吊销",
    # 责任赔偿
    "赔偿", "责任", "侵权", "违约", "损害", "损失",
    # 权利
    "权利", "义务", "隐私", "名誉", "肖像", "人身",
    # 法律关系
    "合同", "协议", "婚姻", "离婚", "继承", "遗嘱",
    "收养", "抚养", "赡养", "劳动", "社保", "工伤",
    # 诉讼
    "诉讼", "仲裁", "起诉", "上诉", "判决", "裁定", "执行",
    "证据", "时效", "管辖", "法院",
    # 犯罪
    "犯罪", "罪名", "故意", "过失", "自首", "累犯",
    "盗窃", "诈骗", "抢劫", "伤害", "杀人",
    # 法律名称简称
    "民法典", "刑法", "宪法", "公司法", "劳动法",
    "治安管理", "道路交通", "行政法", "刑事法",
    # 法律问句模式
    "怎么罚", "判多久", "合法吗", "违法吗", "要不要赔",
    "能告吗", "算不算", "有没有责任",
]


# ---------------------------------------------------------------------------
# Prompt 注入防御：输入安全过滤
# ---------------------------------------------------------------------------

# Prompt 注入攻击模式（命中任一即拒绝请求）
_INJECTION_PATTERNS = [
    # System prompt 泄露 / 越狱
    r'忽略.*(指令|规则|限制|prompt|system|提示)',
    r'(ignore|forget|disregard).*(instruction|rule|prompt|system)',
    r'你.*(是|现在|扮演|作为).*(一个|新的).*(角色|身份)',
    r'DAN\b|jailbreak|越狱',
    r'(print|show|display|reveal|输出).*(system.?prompt|instructions|提示词|系统指令)',
    r'repeat\s+(after\s+me|the\s+following|this)',
    # 角色劫持
    r'(现在开始|从现在起|从今以后).*(你是|你叫|你变成)',
    r'你不再是.*你是',
    r'forget\s+(all|everything).*(before|previous|above)',
    # Token 泄露
    r'(api.?key|secret|token|password|密码).*(告诉我|给我|显示|输出|是什么)',
    r'(what|where)\s+is\s+(your|the)\s+(api.?key|token|secret)',
]

# 敏感内容关键词（涉黄涉政，命中后拒绝服务）
_SENSITIVE_KEYWORDS = [
    # 政治敏感
    "习近平", "江泽民", "胡锦涛", "六四", "天安门", "法轮功",
    "台独", "藏独", "疆独", "港独",
    # 色情暴力
    "色情", "淫秽", "裸体", "性交", "强奸", "杀人方法",
]

# 拒绝回复模板
_INJECTION_REJECT_MSG = "该问题不在我的服务范围内，请提出合法的法律咨询问题。"
_SENSITIVE_REJECT_MSG = "该问题不在我的服务范围内。"


def sanitize_input(query: str) -> tuple[str, bool, str | None]:
    """输入安全过滤，返回 (清洗后文本, 是否安全, 拒绝原因)

    Args:
        query: 原始用户输入

    Returns:
        (清洗后文本, 是否安全, 拒绝原因或None)
        - 安全: 返回原文本 + True + None
        - 不安全: 返回拒绝消息 + False + 拒绝原因
    """
    if not query or not query.strip():
        return query, True, None

    q = query.strip()

    # 1. Prompt 注入检测
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            logger.warning(f"[安全] Prompt 注入拦截: pattern={pattern}, query_preview={q[:100]}")
            return _INJECTION_REJECT_MSG, False, "prompt_injection"

    # 2. 长度限制（防止资源耗尽攻击）
    if len(q) > 2000:
        logger.warning(f"[安全] 输入过长被截断: len={len(q)}")
        q = q[:2000]

    # 3. 敏感内容检测
    nq = _normalize(q)
    for kw in _SENSITIVE_KEYWORDS:
        if _normalize(kw) in nq:
            logger.warning(f"[安全] 敏感词拦截: keyword={kw}, query_preview={q[:100]}")
            return _SENSITIVE_REJECT_MSG, False, "sensitive_content"

    return q, True, None


def _normalize(text: str) -> str:
    """标准化：去标点、去空格、小写"""
    return re.sub(r'[^\w\u4e00-\u9fff]', '', text.lower().strip())


# ---------------------------------------------------------------------------
# 闲聊判定（正则，供 RAGEngine）
# ---------------------------------------------------------------------------

def is_casual_query(query: str) -> bool:
    """快速正则判断是否为明显的闲聊/问候（用于响应元数据标记）"""
    q = query.strip().lower()
    if not q:
        return True
    for pattern in _CASUAL_PATTERNS:
        if re.match(pattern, q):
            return True
    return False


# ---------------------------------------------------------------------------
# 意图分类（关键词，供 LangGraph Agent）
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> bool:
    """意图识别：是否为法律相关问题？

    1. 正则兜底：明显闲聊 → 闲聊
    2. 标准化后精确匹配闲聊短语 → 闲聊
    3. 标准化后包含法律关键词 → 法律
    4. 短查询（≤4字）包含闲聊短语 → 闲聊（二次检查）
    5. 都不匹配 → 默认检索（宁可多检）
    """
    q = query.strip()
    nq = _normalize(q)

    if q and is_casual_query(query):
        return False

    for phrase in _CASUAL_PHRASES:
        if _normalize(phrase) == nq:
            return False

    for kw in _LEGAL_KEYWORDS:
        if _normalize(kw) in nq:
            return True

    if len(nq) <= 4:
        for phrase in _CASUAL_PHRASES:
            if _normalize(phrase) in nq:
                return False

    return True


# 案例/判例关键词 — 命中则分类为案例查询
# 注：匹配在 _normalize 后做子串查找，不支持正则。
# 因此复合模式（如"有没有{任意}案子"）需拆分为独立原子关键词，
# 任一命中即判定为 case_query。
_CASE_KEYWORDS = [
    # 直接指代
    "案例", "判例", "判决书", "判决", "类案", "先例",
    # 组合 — 案例查询常见前缀/中缀
    "指导案例", "典型案件", "最高法案例",
    # 用户口语提问模式（原子化拆分以覆盖"有没有{任意}案子"等句式）
    "有没有",      # 覆盖：有没有{类似盗窃的}案子
    "类似",        # 覆盖：类似{抢劫}的案子、类似{打人}怎么判
    "类似案件",    # 覆盖面兜底
    "怎么判的",    # 覆盖：{这种情况}怎么判的
    "怎么判",      # 覆盖：{盗窃}怎么判
    "有什么案例",  # 覆盖：有什么{相关}案例
    # 三大诉讼类型
    "刑事案件", "民事案件", "行政案件",
    # 口语动作
    "打官司", "翻案",
]


def classify_query_type(query: str) -> str:
    """意图三分类：返回查询类型

    Returns:
        "casual"       — 闲聊/问候，不检索
        "case_query"   — 案例查询，走案例检索路由
        "law_lookup"   — 法律条文查询，走法条检索路由
    """
    q = query.strip()
    nq = _normalize(q)

    # 0. 安全过滤失败 → 特殊处理
    _, is_safe, _ = sanitize_input(q)
    if not is_safe:
        return "casual"

    # 1. 闲聊检测
    if not classify_intent(q):
        return "casual"

    # 2. 案例关键词检测（已去重 normalize，直接在 nq 上匹配）
    for kw in _CASE_KEYWORDS:
        if _normalize(kw) in nq:
            return "case_query"

    # 3. 包含法律关键词 → 法条查询
    for kw in _LEGAL_KEYWORDS:
        if _normalize(kw) in nq:
            return "law_lookup"

    # 4. 默认：长查询走法条检索
    return "law_lookup"


# ---------------------------------------------------------------------------
# LLM 自省路由：判断是否需要检索
# ---------------------------------------------------------------------------

ROUTE_PROMPT = """判断以下用户消息是否需要用法律知识库检索来回答。

## 规则
- YES: 涉及法律条文、法规、处罚、程序、权利等法律专业知识
- NO: 问候、感谢、告别、自我介绍、纯闲聊、日常对话

只输出 YES 或 NO，不要解释。

用户消息: {query}"""


def needs_retrieval(query: str, llm) -> bool:
    """LLM 自省：是否需要检索法律知识库？

    1. 正则命中 → 明确闲聊，不检索
    2. 问题超过 8 个字 → 大概率法律问题，直接检索（零延迟）
    3. 短模糊查询 → LLM 自省判断（正则误杀和真实闲聊的中间地带）
    """
    if is_casual_query(query):
        return False

    # 长查询大概率是正经问题，不走 LLM 路由省一次调用
    if len(query.strip()) > 8:
        return True

    # 短模糊查询：LLM 判断
    prompt = ROUTE_PROMPT.format(query=query)
    result = llm.chat(
        prompt,
        system_prompt="你是一个查询路由判断器。只输出 YES 或 NO。",
    ).strip().upper()

    if "NO" in result:
        return False
    return True
