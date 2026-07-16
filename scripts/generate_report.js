const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  PageBreak, LevelFormat, Header, Footer, PageNumber,
  TableOfContents
} = require('docx');

const COLOR_PRIMARY = "1B3A5C";
const COLOR_ACCENT = "2E75B6";
const COLOR_LIGHT = "D5E8F0";
const COLOR_MUTED = "666666";
const COLOR_GREEN = "1B7A3D";
const COLOR_RED = "C0392B";
const COLOR_ORANGE = "D4A017";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: COLOR_PRIMARY, type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, color: "FFFFFF", font: "Arial", size: 20 })] })]
  });
}

function dataCell(text, width, opts = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.highlight ? { fill: "F0F7E6", type: ShadingType.CLEAR } : (opts.warn ? { fill: "FFF3CD", type: ShadingType.CLEAR } : undefined),
    margins: cellMargins,
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, font: "Arial", size: 19, color: opts.color || "333333", bold: opts.bold })]
    })]
  });
}

function sectionTitle(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 200 }, children: [new TextRun({ text, font: "Arial", bold: true, size: 32, color: COLOR_PRIMARY })] });
}

function subsectionTitle(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, children: [new TextRun({ text, font: "Arial", bold: true, size: 26, color: COLOR_ACCENT })] });
}

function bodyText(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 }, indent: opts.indent ? { left: 360 } : undefined,
    children: [new TextRun({ text, font: "Arial", size: 21, color: "333333", bold: opts.bold, italics: opts.italics })]
  });
}

function bulletItem(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Arial", size: 21, color: "333333" })]
  });
}

function numberItem(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "numbers", level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Arial", size: 21, color: "333333" })]
  });
}

function emptyLine() {
  return new Paragraph({ spacing: { before: 0, after: 0 }, children: [new TextRun({ text: "", font: "Arial", size: 12 })] });
}

function makeTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({ children: headers.map((h, i) => headerCell(h, colWidths[i])) });
  const dataRows = rows.map((row, ri) =>
    new TableRow({ children: row.map((cell, ci) => dataCell(cell, colWidths[ci], {})) })
  );
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

// ========== BUILD DOCUMENT ==========

const children = [];

// ---- COVER PAGE ----
children.push(emptyLine(), emptyLine(), emptyLine(), emptyLine());
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({ text: "Law-RAG-Agent", font: "Arial", bold: true, size: 56, color: COLOR_PRIMARY })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [new TextRun({ text: "基于本地 LLM 的法律法规智能问答系统", font: "Arial", size: 28, color: COLOR_ACCENT })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
  children: [new TextRun({ text: "中期进度汇报", font: "Arial", bold: true, size: 36, color: COLOR_PRIMARY })]
}));
children.push(emptyLine(), emptyLine());
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "汇报日期：2025 年 7 月 15 日", font: "Arial", size: 20, color: COLOR_MUTED })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: "版本：v0.2.0", font: "Arial", size: 20, color: COLOR_MUTED })]
}));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- TABLE OF CONTENTS ----
children.push(sectionTitle("目  录"));
children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ========== SECTION 1: 项目概况 ==========
children.push(sectionTitle("一、项目概况"));

children.push(subsectionTitle("1.1 项目定位"));
children.push(bodyText("Law-RAG-Agent 是一个基于本地大语言模型（LLM）的法律法规智能问答系统。项目整合了检索增强生成（RAG）技术与多 Agent 任务调度框架，以 30 部中国法律原文（共 4,145 条条文）为知识库，为用户提供专业、准确的法律条文查询与解读服务。"));

children.push(subsectionTitle("1.2 技术栈"));
children.push(makeTable(
  ["层级", "技术选型", "说明"],
  [
    ["前端", "Vue 3 + Vite 5 + Pinia", "单页应用，SSE 流式渲染"],
    ["后端框架", "Python 3.12 + FastAPI", "RESTful + SSE 流式接口"],
    ["LLM", "Qwen2.5:7B (Ollama)", "本地部署，零网络依赖"],
    ["Embedding", "BAAI/bge-m3 (Ollama)", "中文法律文本语义检索"],
    ["向量库", "FAISS (CPU)", "3,753 条向量索引"],
    ["Agent 框架", "LangGraph 1.2", "多节点图编排"],
    ["数据库", "PostgreSQL 17 + pgvector", "会话持久化 + 向量检索备选"],
    ["Reranker", "bge-reranker-v2-m3 (Cross-Encoder)", "可选，二次精排"],
    ["包管理", "uv", "Python 依赖管理"],
  ],
  [2200, 2600, 4560]
));
children.push(emptyLine());

children.push(subsectionTitle("1.3 项目规模"));
children.push(makeTable(
  ["指标", "数值"],
  [
    ["Python 源代码行数", "5,363 行（26 个文件）"],
    ["前端代码行数", "约 1,334 行（15 个文件）"],
    ["后端模块数", "6 个（api, agents, rag, llm, embedding, chunking）"],
    ["前端组件数", "5 个（LoginView, ChatView, Sidebar, ChatMessage, ChatInput）"],
    ["API 端点", "7 个（health, chat, chat/stream, auth×3, conversations×3）"],
    ["Git 提交次数", "22 次"],
    ["法律数据库", "30 部中国法律, 4,145 条原文"],
    ["FAISS 索引量", "3,753 条向量（law_index_bge）"],
  ],
  [4000, 5360]
));
children.push(emptyLine());

children.push(subsectionTitle("1.4 当前运行配置"));
children.push(makeTable(
  ["配置项", "状态", "说明"],
  [
    ["AGENT_ENABLED", "✓ 开启", "LangGraph 多 Agent（改写+检索+生成+审核）"],
    ["RERANK_ENABLED", "✗ 关闭", "CPU 推理太慢（20-30s），暂不启用"],
    ["ADJACENT_ENABLED", "✓ 开启", "±2 条相邻法条扩展"],
    ["PG_ENABLED", "✗ 关闭", "使用 FAISS 本地索引"],
    ["LLM_MODEL", "qwen2.5:7b", "Ollama 本地推理"],
    ["EMBED_MODEL", "bge-m3", "Ollama Embedding API"],
  ],
  [3200, 1600, 4560]
));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ========== SECTION 2: 核心功能实现 ==========
children.push(sectionTitle("二、核心功能实现"));

children.push(subsectionTitle("2.1 Agent 图（LangGraph 多节点编排）"));
children.push(bodyText("项目核心采用 LangGraph 构建多 Agent 工作流图，实现查询改写、语义检索、答案生成、质量审核的完整闭环。"));
children.push(emptyLine());

children.push(bodyText("图结构（运行时流式执行）：", { bold: true }));
children.push(emptyLine());
children.push(bodyText("  intent（意图识别）", { indent: true }));
children.push(bodyText("    ├─ 闲聊 → casual_reply → LLM 直回 → END", { indent: true }));
children.push(bodyText("    └─ 法律 → rewrite → retrieve → generate → validate", { indent: true }));
children.push(bodyText("                                        ↑                │", { indent: true }));
children.push(bodyText("                                        └── FAIL 重试 ────┘", { indent: true }));
children.push(emptyLine());

children.push(bodyText("六个核心节点：", { bold: true }));
children.push(bulletItem("intent — 意图识别节点。基于关键字匹配（70+ 法律关键词 + 40+ 闲聊短语），标准化去标点后精确匹配，识别用户是闲聊还是法律问题。"));
children.push(bulletItem("casual_reply — 闲聊回复节点。直接调用 LLM 回答问候/感谢/告别等非法律问题，不触发检索。"));
children.push(bulletItem("rewrite — 查询改写节点。将用户自然语言问题改写为正式法律术语，补充不完整的法律名称，提高检索命中率。"));
children.push(bulletItem("retrieve — 检索节点。调用 FAISS 向量库进行语义检索（top_k=5），并应用相邻法条扩展（±2 条）保证条文连续性。"));
children.push(bulletItem("generate — 生成节点。构建层级结构化 Prompt（法律→章→节→条文），逐 token 流式输出 LLM 回答，无 <think> 标签污染。"));
children.push(bulletItem("validate — 审核节点。将检索条文、用户问题、AI 回答一并提交 LLM 审核，检查事实性幻觉（虚构法条、篡改条文）。FAIL 时最多重试 1 次。"));

children.push(subsectionTitle("2.2 意图识别模块"));
children.push(bodyText("意图识别是 Agent 的第一道关口，决定查询走\"闲聊直回\"还是\"法律检索\"路径。采用四层纯关键字匹配，零 LLM 调用延迟："));
children.push(emptyLine());

children.push(bodyText("第一层：标准化后精确匹配闲聊短语", { bold: true }));
children.push(bodyText("  _normalize() 去标点、去空格、小写 → \"你好！\" 标准化为 \"你好\"", { indent: true }));
children.push(bodyText("  命中即返回闲聊（False），不触发检索。", { indent: true }));
children.push(emptyLine());
children.push(bodyText("第二层：标准化后包含法律关键词", { bold: true }));
children.push(bodyText("  命中任一关键词（\"处罚\"\"合同\"\"诉讼\"\"合法吗\"等 70+ 个）→ 走检索。", { indent: true }));
children.push(emptyLine());
children.push(bodyText("第三层：短查询（≤4 字）二次闲聊检查", { bold: true }));
children.push(bodyText("  对\"嗯\"\"哦\"等极短输入做包含匹配，防止误判为法律问题。", { indent: true }));
children.push(emptyLine());
children.push(bodyText("第四层：默认走检索", { bold: true }));
children.push(bodyText("  \"宁可多检索十条，不漏一个法律问题。\"", { indent: true }));

children.push(subsectionTitle("2.3 流式输出与思考过程展示"));
children.push(bodyText("Agent 流式接口（SSE）按顺序输出以下事件类型："));
children.push(emptyLine());
children.push(makeTable(
  ["事件类型", "内容示例", "前端渲染位置"],
  [
    ["thinking", "🔧 正在初始化 Agent...", "思考框"],
    ["thinking", "🎯 意图识别: 法律问题 → 检索法条", "思考框"],
    ["thinking", "📝 查询改写: xxx", "思考框"],
    ["thinking", "🔍 正在检索法律条文...", "思考框"],
    ["meta", "{sources: [...], is_casual: false}", "元数据"],
    ["thinking", "💭 模型正在思考...", "思考框"],
    ["token", "根据治安管理处罚法...", "回答框（流式）"],
    ["thinking", "🔎 审核回答质量... → ✅ 审核通过", "思考框"],
  ],
  [2000, 4200, 3160]
));
children.push(emptyLine());

children.push(bodyText("思考过程持久化：Assistant 消息携带 thinking 字段存入 PostgreSQL conversations 表 JSONB 列，切换对话后自动恢复，新建对话自动清空。"));

children.push(subsectionTitle("2.4 答案审核机制"));
children.push(bodyText("审核节点（validate）是防幻觉的最后一道防线。LLM 同时接收\"检索到的法律条文 + 用户问题 + AI 回答\"三项信息，判断回答中是否存在以下幻觉类型："));
children.push(bulletItem("虚构、篡改或引用不存在的法律条文"));
children.push(bulletItem("将已失效的法律当作现行有效法律使用"));
children.push(bulletItem("法律原则、法律后果、程序规则的根本性错误"));
children.push(bulletItem("关键事实假设毫无根据，严重误导用户"));
children.push(bodyText("审核不通过时，系统自动向生成节点反馈错误原因（\"回答未引用法律名称或条款号\"），触发最多 1 次重新生成。"));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ========== SECTION 3: 基础设施 ==========
children.push(sectionTitle("三、基础设施建设"));

children.push(subsectionTitle("3.1 用户认证与会话隔离"));
children.push(bodyText("项目实现了完整的用户认证体系："));
children.push(bulletItem("PBKDF2-SHA256 密码哈希存储（users 表）"));
children.push(bulletItem("Bearer Token 鉴权（SHA256 哈希存储，服务端校验）"));
children.push(bulletItem("用户名+密码注册/登录，返回 Token"));
children.push(bulletItem("会话按 user_id 隔离，每个用户只能访问自己的对话记录"));
children.push(bulletItem("支持匿名用户回退机制（未登录时可有限使用）"));
children.push(bulletItem("启动时 Token 缓存加载，重启后 Token 仍有效"));

children.push(subsectionTitle("3.2 会话持久化"));
children.push(bodyText("采用 PostgreSQL JSONB 存储方案，每个会话一行记录："));
children.push(bulletItem("conversations 表：user_id + session_id + messages(JSONB) + 时间戳"));
children.push(bulletItem("messages 为完整消息数组 [{role, content, thinking?}, ...]"));
children.push(bulletItem("UNIQUE (user_id, session_id) 索引，ON CONFLICT UPSERT"));
children.push(bulletItem("列表接口按 updated_at 倒序，展示首条用户消息作为摘要"));
children.push(bulletItem("思考过程作为 thinking 字段嵌入 assistant 消息持久化"));

children.push(subsectionTitle("3.3 异常处理体系"));
children.push(bodyText("建立了三层异常处理机制："));
children.push(emptyLine());
children.push(makeTable(
  ["层级", "触发条件", "响应"],
  [
    ["全局异常处理器", "TimeoutError", "JSON: {error, detail, code: TIMEOUT}, 504"],
    ["全局异常处理器", "ConnectionError", "JSON: {error, detail, code: SERVICE_UNAVAILABLE}, 503"],
    ["全局异常处理器", "Exception (兜底)", "JSON: {error, detail, code: INTERNAL_ERROR}, 500"],
    ["路由层 try/except", "/chat 异常", "HTTPException 500"],
    ["流式 SSE error", "stream generate() 异常", "SSE: {type: error, content}"],
    ["LLM 调用超时", "ollama.Client(timeout=300s)", "TimeoutError → 全局处理器"],
  ],
  [2200, 3100, 4060]
));
children.push(emptyLine());

children.push(subsectionTitle("3.4 PostgreSQL 自动重连"));
children.push(bodyText("ConversationStore 和 PgvectorRetriever 均实现了 _ensure_connection() 方法："));
children.push(bulletItem("每次操作前执行 SELECT 1 探活"));
children.push(bulletItem("连接断开时自动关闭旧连接并重建"));
children.push(bulletItem("PgvectorRetriever 重连后重新 register_vector()"));
children.push(bulletItem("PostgreSQL 重启后无需手动重启应用服务"));

children.push(subsectionTitle("3.5 前端架构"));
children.push(bodyText("Vue 3 + Vite 5 单页应用，核心组件："));
children.push(bulletItem("LoginView — 登录/注册页面"));
children.push(bulletItem("ChatView — 主对话页面（集成思考框 + 消息列表 + 输入框）"));
children.push(bulletItem("Sidebar — 可收纳侧边对话栏，显示历史会话列表"));
children.push(bulletItem("ChatMessage — 消息卡片组件（Markdown 渲染）"));
children.push(bulletItem("ChatInput — 输入组件（发送/清空/检索数量调节）"));
children.push(bulletItem("Pinia 状态管理：auth store（认证）+ chat store（对话）"));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ========== SECTION 4: 数据管线 ==========
children.push(sectionTitle("四、数据管线"));

children.push(subsectionTitle("4.1 法律文档处理"));
children.push(bodyText("30 部中国法律原文经过完整的 ETL 管线处理："));
children.push(bulletItem("爬取 → 结构解析（parser.py, 405 行）→ 智能切分（chunker.py, 287 行）"));
children.push(bulletItem("切分策略：以\"条\"为最小单元，短于 50 字自动合并"));
children.push(bulletItem("每章生成摘要 chunk，保留 章/节/条 层级结构"));
children.push(bulletItem("bge-m3 向量化 → FAISS 索引构建 → 3,753 条向量"));
children.push(bulletItem("相邻法条映射表（article_map.json）→ 检索后 ±N 条扩展"));
children.push(bulletItem("BM25 索引备选构建（默认关闭，bge-m3 语义检索已足够）"));

children.push(subsectionTitle("4.2 Prompt 工程设计"));
children.push(makeTable(
  ["Prompt", "用途", "关键要求"],
  [
    ["RAG_PROMPT_TEMPLATE", "法律条文问答生成", "引用法律名称+条款号，不编造，诚实说明不足"],
    ["REWRITE_PROMPT", "查询改写", "保留核心概念，补全法律名称，正式术语"],
    ["VALIDATOR_PROMPT", "答案幻觉审核", "输入：问题+条文+回答，输出：PASS/FAIL+理由"],
    ["ROUTE_PROMPT", "LLM 路由判断（非 Agent）", "判断是否需要检索"],
    ["CASUAL_SYSTEM_PROMPT", "闲聊回复", "友好自然，引导用户提出法律问题"],
  ],
  [3000, 2560, 3800]
));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ========== SECTION 5: 当前问题与待解决 ==========
children.push(sectionTitle("五、当前问题与待解决事项"));

children.push(subsectionTitle("5.1 功能性问题"));

children.push(makeTable(
  ["优先级", "问题", "影响", "建议"],
  [
    ["P1-高", "Reranker CPU 推理过慢", "开启后检索延迟 20-30 秒，严重影响用户体验", "短期关闭；长期考虑 GPU 加速或轻量级 Reranker"],
    ["P1-高", "非 Agent 模式路由不一致", "流式接口用了 LLM 路由，同步接口未同步", "已部分修复，需确保两个接口逻辑一致"],
    ["P2-中", "Agent 流式非真流式（LLM token 延迟）", "用户感知不够流畅", "当前 Per-token 冲洗已基本解决"],
    ["P2-中", "前端 saveSession 无错误处理", "保存失败时用户无感知", "添加保存失败提示和重试逻辑"],
    ["P2-中", "Vue Router 缺导航守卫", "未登录用户刷新页面短暂看到未授权页面", "添加 router.beforeEach"],
    ["P2-中", "ChatView clear 逻辑冗余", "indexOf 引用比较不稳定", "简化 while loop 逻辑"],
  ],
  [1200, 3160, 2000, 3000]
));
children.push(emptyLine());

children.push(subsectionTitle("5.2 性能优化空间"));
children.push(bulletItem("Reranker 启用时延迟过高（20-30s）— 需要 GPU 或模型量化"));
children.push(bulletItem("FAISS 索引首次加载耗时（~3s, 3753 条向量）— 初次请求感知延迟"));
children.push(bulletItem("LLM 冷启动慢（Ollama 首次加载模型 ~10-30s）— 可考虑 lifespan 预热"));
children.push(bulletItem("Embedding 批量处理已优化（batch_size=32），无进一步优化空间"));
children.push(bulletItem("检索结果层级结构化 Prompt — token 消耗较高，可适当精简"));

children.push(subsectionTitle("5.3 工程化待完善"));
children.push(bulletItem("单元测试覆盖不足 — 当前仅有 scripts/ 脚本级测试，缺乏 pytest 自动化测试"));
children.push(bulletItem("CI/CD 缺失 — 无自动化构建/测试/部署流水线"));
children.push(bulletItem("日志体系不完善 — print() 和 logger 混用，需统一升级"));
children.push(bulletItem("配置管理 — PG_CONN 等多个值在多处重复定义"));
children.push(bulletItem("Docker Compose 环境变量不完整 — 部分配置需通过 .env 文件补充"));
children.push(bulletItem("前端存在两个版本 — static/index.html（原生）和 frontend/（Vue SPA）需二选一"));
children.push(bulletItem("Vite 构建输出路径与 FastAPI 静态文件挂载不匹配"));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ========== SECTION 6: 后续计划 ==========
children.push(sectionTitle("六、后续开发计划"));

children.push(subsectionTitle("6.1 短期（1-2 周）"));
children.push(numberItem("性能优化 — 解决 Reranker 延迟问题（模型量化 / 可选 GPU 加速）"));
children.push(numberItem("前端完善 — 修复 saveSession 错误处理、添加路由守卫"));
children.push(numberItem("代码清理 — 移除 static/index.html，统一使用 Vue 前端"));
children.push(numberItem("统一路由逻辑 — 确保 Agent/非 Agent 模式路由行为一致"));
children.push(numberItem("日志体系升级 — 统一使用 logger，添加请求追踪"));

children.push(subsectionTitle("6.2 中期（2-4 周）"));
children.push(numberItem("知识图谱集成 — 法律条文关联关系（引用/上位法/下位法）"));
children.push(numberItem("多轮对话增强 — 上下文感知的查询改写和意图追踪"));
children.push(numberItem("检索效果评估 — 完善 eval_retrieval.py 中的 Precision/NDCG 计算"));
children.push(numberItem("用户反馈机制 — 点赞/踩 + 反馈理由收集"));
children.push(numberItem("多模型支持 — 支持切换不同 LLM / Embedding 模型"));

children.push(subsectionTitle("6.3 长期（1-2 月）"));
children.push(numberItem("生产化部署 — Docker 容器化完善 + Nginx 反向代理"));
children.push(numberItem("监控告警 — 请求延迟、错误率、LLM 调用量监控"));
children.push(numberItem("权限管理 — 多角色用户（管理员/普通用户/访客）"));
children.push(numberItem("法律更新机制 — 新法发布自动更新知识库"));
children.push(numberItem("API 文档完善 — OpenAPI/Swagger 文档补充 + 示例代码"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ========== SECTION 7: 附录 ==========
children.push(sectionTitle("七、附录"));

children.push(subsectionTitle("7.1 项目文件结构"));
children.push(bodyText("src/", { bold: true }));
children.push(bodyText("  api/       — FastAPI 应用入口、路由、认证、依赖注入 (main.py, routes.py, auth.py, dependencies.py, models.py, conversation_store.py)", { indent: true }));
children.push(bodyText("  agents/    — LangGraph 多 Agent 图编排 (graph.py, 527行)", { indent: true }));
children.push(bodyText("  rag/       — RAG 引擎、检索器、混合检索、重排序、相邻扩展 (engine.py, retriever.py, hybrid_retriever.py, reranker.py, adjacent_expander.py)", { indent: true }));
children.push(bodyText("  llm/       — Ollama LLM 客户端 (client.py, 373行)", { indent: true }));
children.push(bodyText("  embedding/ — 向量化模块、FAISS 向量库 (embedder.py, vector_store.py)", { indent: true }));
children.push(bodyText("  chunking/  — 法律文档结构解析和智能切分 (parser.py, chunker.py)", { indent: true }));
children.push(bodyText("  config.py  — 统一配置模块 (92行)", { indent: true }));
children.push(emptyLine());

children.push(bodyText("frontend/ — Vue 3 + Vite 5 单页应用", { bold: true }));
children.push(bodyText("  src/views/          — ChatView.vue (300行), LoginView.vue (164行)", { indent: true }));
children.push(bodyText("  src/components/     — Sidebar, ChatMessage, ChatInput", { indent: true }));
children.push(bodyText("  src/stores/         — Pinia 状态：auth.js, chat.js", { indent: true }));
children.push(bodyText("  src/api/            — fetch 封装 + SSE 流式处理", { indent: true }));
children.push(emptyLine());

children.push(bodyText("scripts/ — 工具脚本（8 个）", { bold: true }));
children.push(bodyText("  build_index.py, eval_retrieval.py, test_rag.py, test_llm.py, compare_models.py, batch_eval.py, migrate_to_pgvector.py, build_bm25.py", { indent: true }));
children.push(emptyLine());

children.push(subsectionTitle("7.2 API 接口一览"));
children.push(makeTable(
  ["端点", "方法", "功能", "认证"],
  [
    ["/api/health", "GET", "健康检查（LLM 模型、索引状态、文档数）", "否"],
    ["/api/auth/register", "POST", "用户注册", "否"],
    ["/api/auth/login", "POST", "用户登录，返回 Bearer Token", "否"],
    ["/api/auth/me", "GET", "获取当前用户信息", "是"],
    ["/api/chat", "POST", "同步问答（支持 Agent / 非 Agent）", "否"],
    ["/api/chat/stream", "POST", "SSE 流式问答（思考过程+回答）", "否"],
    ["/api/conversations", "GET", "列出当前用户会话列表", "是"],
    ["/api/conversations/{id}", "GET", "加载指定会话历史", "是"],
    ["/api/conversations/{id}", "POST", "保存/更新会话消息", "是"],
  ],
  [3600, 1000, 3360, 1400]
));
children.push(emptyLine());

children.push(subsectionTitle("7.3 技术债务清单"));
children.push(makeTable(
  ["类别", "债务项", "紧迫度"],
  [
    ["代码质量", "ChatView clear 逻辑冗余", "低"],
    ["代码质量", "print() 与 logger 混用（scripts/）", "中"],
    ["架构", "static/index.html 与 frontend/ 并存", "中"],
    ["架构", "Vite 构建输出路径不匹配（static-vue vs static）", "中"],
    ["测试", "缺少自动化测试（0 个 unittest/pytest）", "高"],
    ["安全", "Token 存 localStorage（本地部署无影响，生产需改 httpOnly Cookie）", "低"],
    ["配置", "PG_CONN 等多处置复定义默认值", "低"],
    ["性能", "Reranker CPU 推理 20-30s", "高"],
  ],
  [1600, 5360, 2400]
));

// Build document
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1200, bottom: 1440, left: 1200 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: COLOR_PRIMARY, space: 4 } },
          children: [new TextRun({ text: "Law-RAG-Agent 中期进度汇报", font: "Arial", size: 16, color: COLOR_MUTED, italics: true })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          border: { top: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC", space: 4 } },
          children: [new TextRun({ text: "第 ", font: "Arial", size: 16, color: COLOR_MUTED }), new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: COLOR_MUTED }), new TextRun({ text: " 页", font: "Arial", size: 16, color: COLOR_MUTED })]
        })]
      })
    },
    children
  }]
});

const OUTPUT = "docs/Law-RAG-Agent_中期进度汇报.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUTPUT, buf);
  console.log("Report generated: " + OUTPUT);
});
