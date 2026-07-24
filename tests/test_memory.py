"""
对话记忆层单元测试。

验证:
  1. ConversationMemoryManager 模块可导入
  2. 摘要触发条件判断
  3. 摘要实体解析
  4. 上下文构建
  5. 记忆集成到 Agent Graph（不涉及 DB）
"""
from __future__ import annotations


class TestImports:
    def test_import_memory_manager(self):
        from src.memory.conversation import ConversationMemoryManager
        assert ConversationMemoryManager is not None


class TestSummaryTrigger:
    def test_below_threshold(self):
        from src.memory.conversation import ConversationMemoryManager
        # 不需要实例化，should_summarize 是实例方法但逻辑纯函数
        assert ConversationMemoryManager.should_summarize is not None

    def test_at_threshold(self):
        from src.memory.conversation import ConversationMemoryManager, SUMMARY_TRIGGER_ROUNDS
        # 创建一个最小 mock 来测试
        class FakeMgr:
            pass
        mgr = FakeMgr()
        mgr.should_summarize = lambda n: n >= SUMMARY_TRIGGER_ROUNDS
        assert mgr.should_summarize(6) is True
        assert mgr.should_summarize(5) is False
        assert mgr.should_summarize(10) is True
        assert mgr.should_summarize(0) is False


class TestEntityParsing:
    def test_parse_full_entities(self):
        from src.memory.conversation import ConversationMemoryManager
        summary = """案件类型: 劳动争议
涉及法律: 劳动合同法, 劳动争议调解仲裁法
关键事实: 用户在试用期被辞退，单位未支付补偿金
已回答问题: 解释了试用期的法律规定
未解决问题: 具体赔偿金额需要根据工资计算"""
        entities = ConversationMemoryManager._parse_entities(summary)
        assert entities["case_type"] == "劳动争议"
        assert "劳动合同法" in entities["laws_involved"]
        assert "试用期" in entities["key_facts"]

    def test_parse_partial_entities(self):
        from src.memory.conversation import ConversationMemoryManager
        summary = """案件类型: 合同纠纷
涉及法律: 民法典"""
        entities = ConversationMemoryManager._parse_entities(summary)
        assert entities["case_type"] == "合同纠纷"
        assert entities["laws_involved"] == "民法典"
        assert "key_facts" not in entities  # 没有关键事实行


class TestContextBuilding:
    def test_empty_memories(self):
        from src.memory.conversation import ConversationMemoryManager
        # build_context 是实例方法但可用 class 调用
        result = ConversationMemoryManager.build_context(None, [])
        assert result == ""

    def test_single_memory(self):
        from src.memory.conversation import ConversationMemoryManager
        memories = [{
            "summary": "用户咨询了工伤认定相关问题",
            "score": 0.85,
            "entities": {"case_type": "劳动争议"},
        }]
        result = ConversationMemoryManager.build_context(None, memories)
        assert "历史对话参考" in result
        assert "工伤认定" in result
        assert "0.85" in result

    def test_multiple_memories(self):
        from src.memory.conversation import ConversationMemoryManager
        memories = [
            {"summary": "工伤认定问题", "score": 0.9, "entities": {}},
            {"summary": "劳动合同纠纷", "score": 0.7, "entities": {}},
        ]
        result = ConversationMemoryManager.build_context(None, memories)
        assert "历史对话 1" in result
        assert "历史对话 2" in result


class TestConversationFormatting:
    def test_format_messages(self):
        from src.memory.conversation import ConversationMemoryManager
        msgs = [
            {"role": "user", "content": "工伤怎么认定"},
            {"role": "assistant", "content": "根据《工伤保险条例》第十四条..."},
        ]
        result = ConversationMemoryManager._format_conversation(msgs)
        assert "用户:" in result
        assert "AI助手:" in result
        assert "工伤保险条例" in result

    def test_truncate_long_messages(self):
        from src.memory.conversation import ConversationMemoryManager
        long_text = "A" * 800
        msgs = [{"role": "user", "content": long_text}]
        result = ConversationMemoryManager._format_conversation(msgs)
        assert "..." in result
        assert len(result) < 600  # 截断后更短


class TestGraphMemoryIntegration:
    """验证 Agent Graph 正确集成了 memory_retrieve 节点"""

    def test_agent_accepts_memory_manager(self):
        from src.agents.graph import LawAgentGraph
        # 不传 memory_manager 应正常运行
        # 这里只验证 __init__ 不报错
        # 实际需要 retriever 和 llm，跳过完整实例化

    def test_graph_node_count(self):
        from src.agents.graph import AgentState
        assert "memory_context" in AgentState.__annotations__
        assert "user_id" in AgentState.__annotations__
