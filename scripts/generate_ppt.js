/**
 * 项目答辩 PPT 生成脚本
 * 用法: node scripts/generate_ppt.js
 * 输出: docs/presentation.pptx (8 页, 16:9)
 */
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Law-RAG-Agent Team";
pres.title = "Law-RAG-Agent 项目答辩";

// ========== 色彩系统 ==========
const C = {
  navy:    "1B2A4A",
  dark:    "0F1A2E",
  gold:    "D4A84B",
  goldLight:"E8D5A3",
  cream:   "F7F3EC",
  white:   "FFFFFF",
  text:    "1E293B",
  muted:   "64748B",
  green:   "059669",
  red:     "DC2626",
  teal:    "0D9488",
};

// ========== 辅助函数 ==========
const makeShadow = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.10 });

function addTitleBar(slide, title) {
  slide.background = { color: C.cream };
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  slide.addText(title, { x: 0.6, y: 0.2, w: 9, h: 0.6, fontSize: 28, fontFace: "Arial", bold: true, color: C.navy, margin: 0 });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.9, w: 10, h: 0.02, fill: { color: C.goldLight } });
}

function addCard(slide, x, y, w, h, icon, title, text, accent) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.white }, shadow: makeShadow() });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h, fill: { color: accent || C.gold } });
  slide.addText(icon, { x: x + 0.25, y: y + 0.12, w: 0.5, h: 0.5, fontSize: 24, align: "center", margin: 0 });
  slide.addText(title, { x: x + 0.8, y: y + 0.12, w: w - 1.1, h: 0.35, fontSize: 14, fontFace: "Arial", bold: true, color: C.navy, margin: 0 });
  slide.addText(text, { x: x + 0.25, y: y + 0.55, w: w - 0.55, h: h - 0.65, fontSize: 11, fontFace: "Arial", color: C.muted, margin: 0 });
}

function addBigNumber(slide, x, y, number, label, color) {
  slide.addText(String(number), { x, y, w: 2, h: 0.7, fontSize: 40, fontFace: "Arial", bold: true, color: color || C.gold, align: "center", margin: 0 });
  slide.addText(label, { x, y: y + 0.62, w: 2, h: 0.4, fontSize: 11, fontFace: "Arial", color: C.muted, align: "center", margin: 0 });
}

function addPageNumber(slide, num) {
  slide.addText(String(num), { x: 9.3, y: 5.2, w: 0.5, h: 0.3, fontSize: 9, fontFace: "Arial", color: C.muted, align: "right" });
}

// ================================================================
// Slide 1: 封面
// ================================================================
(function() {
  const s = pres.addSlide();
  s.background = { color: C.dark };
  // 顶部金线
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  // 装饰性背景块
  s.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 1.8, w: 7, h: 2.4, fill: { color: C.navy }, shadow: makeShadow() });
  // 项目名称
  s.addText("基于本地大语言模型的\n法律法规智能问答系统\n构建与 Agent 任务调度实践", {
    x: 0.8, y: 0.6, w: 8.4, h: 1.8,
    fontSize: 26, fontFace: "Arial", bold: true, color: C.white,
    align: "center", valign: "bottom",
  });
  // 项目简称
  s.addText("Law-RAG-Agent", {
    x: 2, y: 2.6, w: 6, h: 0.8,
    fontSize: 44, fontFace: "Arial Black", color: C.gold, align: "center", charSpacing: 6,
  });
  s.addText("奥马灯塔", {
    x: 2, y: 3.2, w: 6, h: 0.5,
    fontSize: 18, fontFace: "Arial", color: C.goldLight, align: "center",
  });
  // 底部信息
  s.addText("实习项目答辩  |  2026 年 7 月", {
    x: 2, y: 3.9, w: 6, h: 0.4,
    fontSize: 12, fontFace: "Arial", color: C.muted, align: "center",
  });
  s.addText("组员：XXX  XXX  XXX  XXX  XXX", {
    x: 2, y: 4.3, w: 6, h: 0.35,
    fontSize: 10, fontFace: "Arial", color: C.muted, align: "center",
  });
})();

// ================================================================
// Slide 2: 项目背景与目标
// ================================================================
(function() {
  const s = pres.addSlide();
  addTitleBar(s, "项目背景与目标");
  addPageNumber(s, 2);

  // 左侧 - 背景
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.2, w: 4, h: 3.8, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.2, w: 0.07, h: 3.8, fill: { color: C.navy } });
  s.addText("▎", { x: 0.15, y: 1.2, w: 0.5, h: 3.8, fontSize: 140, fontFace: "Arial", color: C.navy, margin: 0, transparency: 80 });
  s.addText("背景", { x: 1.0, y: 1.35, w: 3, h: 0.35, fontSize: 16, fontFace: "Arial", bold: true, color: C.navy, margin: 0 });
  s.addText([
    { text: "法律领域知识密度高、更新频繁", options: { bullet: true, breakLine: true, fontSize: 11, color: C.text } },
    { text: "通用大模型存在幻觉与时效性问题", options: { bullet: true, breakLine: true, fontSize: 11, color: C.text } },
    { text: "法律实务需要精准的条文引用", options: { bullet: true, breakLine: true, fontSize: 11, color: C.text } },
    { text: "本地化部署保障数据安全与隐私", options: { bullet: true, breakLine: true, fontSize: 11, color: C.text } },
    { text: "RAG+Agent 结合可提升复杂问答质量", options: { bullet: true, fontSize: 11, color: C.text } },
  ], { x: 1.0, y: 1.85, w: 3.3, h: 3.0, paraSpaceAfter: 8, valign: "top" });

  // 右侧 - 目标
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.2, w: 4.2, h: 3.8, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.2, w: 0.07, h: 3.8, fill: { color: C.gold } });
  s.addText("目标", { x: 5.6, y: 1.35, w: 3, h: 0.35, fontSize: 16, fontFace: "Arial", bold: true, color: C.navy, margin: 0 });
  s.addText([
    { text: "部署本地 LLM，构建端到端 RAG 问答系统", options: { bullet: true, breakLine: true, fontSize: 11, color: C.text } },
    { text: "爬取并向量化 30 部中国法律全文", options: { bullet: true, breakLine: true, fontSize: 11, color: C.text } },
    { text: "实现 LangGraph Agent 多步骤任务调度", options: { bullet: true, breakLine: true, fontSize: 11, color: C.text } },
    { text: "提供 FastAPI + Vue 3 交互界面", options: { bullet: true, breakLine: true, fontSize: 11, color: C.text } },
    { text: "构造 100+ 条标注测试集, 系统评测交付", options: { bullet: true, fontSize: 11, color: C.text } },
  ], { x: 5.6, y: 1.85, w: 3.5, h: 3.0, paraSpaceAfter: 8, valign: "top" });

  // 底部总结
  s.addText("构建一款全本地化、可演示的法律法规智能问答系统原型，通过 RAG + Agent 技术解决法律场景的知识检索与精准问答需求。", {
    x: 0.6, y: 5.15, w: 8.8, h: 0.3, fontSize: 11, fontFace: "Arial", italic: true, color: C.muted,
  });
})();

// ================================================================
// Slide 3: 核心功能与亮点
// ================================================================
(function() {
  const s = pres.addSlide();
  addTitleBar(s, "核心功能与亮点");
  addPageNumber(s, 3);

  // 2x2 卡片布局
  const cards = [
    { icon: "\u{1F50D}", title: "多阶段检索链路", text: "FAISS 向量检索 (bge-m3) \u2192 chunk_type 噪声过滤 \u2192 Cross-Encoder 精排 \u2192 相邻条文扩展, 确保召回 80% 且无章级噪声", accent: C.teal },
    { icon: "\u{1F916}", title: "LangGraph Agent 调度", text: "6 节点状态图: 意图识别 \u2192 查询改写 \u2192 检索 \u2192 生成 \u2192 答案校验 \u2192 重试循环, 支持多轮上下文保持", accent: C.gold },
    { icon: "\u{26A1}", title: "SSE 流式问答", text: "首字延迟 < 1s, 逐字渲染, 前端 Vue 3 实时展示 + 引用来源折叠, 体验对标 ChatGPT", accent: C.green },
    { icon: "\u{1F4CA}", title: "完整评测体系", text: "131 条标注测试集 / 174 单元测试 / 检索 Recall@5=80% / 回答综合评分 0.817 / 真实幻觉率 0%", accent: C.red },
  ];

  for (let i = 0; i < 4; i++) {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const x = 0.6 + col * 4.6;
    const y = 1.2 + row * 2.05;
    addCard(s, x, y, 4.2, 1.85, cards[i].icon, cards[i].title, cards[i].text, cards[i].accent);
  }

  s.addText("四大功能模块协同工作，形成从知识入库到用户交互的完整闭环，评测数据支撑质量可信。", {
    x: 0.6, y: 5.25, w: 8.8, h: 0.25, fontSize: 11, fontFace: "Arial", italic: true, color: C.muted,
  });
})();

// ================================================================
// Slide 4: 技术架构概览
// ================================================================
(function() {
  const s = pres.addSlide();
  addTitleBar(s, "技术架构概览");
  addPageNumber(s, 4);

  // 架构图 - 用分层盒子表示
  const layers = [
    { label: "展示层", items: "Vue 3 + Vite + Pinia  |  Swagger UI", x: 0.8, color: C.teal },
    { label: "服务层", items: "FastAPI + JWT + LangGraph + SSE", x: 0.8, color: C.gold },
    { label: "推理层", items: "Ollama + qwen2.5:7b + bge-m3 + bge-reranker-v2-m3", x: 0.8, color: C.green },
    { label: "存储层", items: "FAISS (3753 文档) + BM25 语料 + PostgreSQL", x: 0.8, color: C.navy },
  ];

  layers.forEach((layer, i) => {
    const y = 1.2 + i * 0.92;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 4.6, h: 0.75, fill: { color: C.white }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y, w: 0.07, h: 0.75, fill: { color: layer.color } });
    s.addText(layer.label, { x: 0.95, y: y + 0.08, w: 1.2, h: 0.25, fontSize: 12, fontFace: "Arial", bold: true, color: C.navy, margin: 0 });
    s.addText(layer.items, { x: 0.95, y: y + 0.35, w: 4, h: 0.3, fontSize: 10, fontFace: "Arial", color: C.muted, margin: 0 });
    // 箭头 (除最后)
    if (i < 3) {
      s.addText("\u25BC", { x: 2.5, y: y + 0.75, w: 0.6, h: 0.2, fontSize: 10, align: "center", color: C.muted });
    }
  });

  // 右侧 - 技术栈清单
  s.addShape(pres.shapes.RECTANGLE, { x: 5.6, y: 1.2, w: 3.8, h: 3.7, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.6, y: 1.2, w: 3.8, h: 0.45, fill: { color: C.navy } });
  s.addText("技术选型", { x: 5.6, y: 1.2, w: 3.8, h: 0.45, fontSize: 14, fontFace: "Arial", bold: true, color: C.white, align: "center", margin: 0 });

  const techs = [
    ["LLM", "Qwen2.5:7b (Ollama)"], ["Embedding", "bge-m3 (1024d)"],
    ["Reranker", "bge-reranker-v2-m3"], ["向量索引", "FAISS IndexFlatIP"],
    ["Agent", "LangGraph 1.2"], ["Web", "FastAPI + Vue 3"],
    ["分词", "jieba"], ["认证", "JWT"],
    ["包管理", "uv"], ["部署", "Docker Compose"],
  ];

  techs.forEach((t, i) => {
    s.addText(t[0], { x: 5.85, y: 1.85 + i * 0.32, w: 1.2, h: 0.25, fontSize: 10, fontFace: "Arial", bold: true, color: C.navy, margin: 0 });
    s.addText(t[1], { x: 7.1, y: 1.85 + i * 0.32, w: 2.1, h: 0.25, fontSize: 9, fontFace: "Arial", color: C.muted, margin: 0 });
  });

  s.addText("四层架构清晰分离，全链路本地部署，Ollama 统一管理 Embedding + LLM + Reranker 三类模型。", {
    x: 0.6, y: 5.15, w: 8.8, h: 0.3, fontSize: 11, fontFace: "Arial", italic: true, color: C.muted,
  });
})();

// ================================================================
// Slide 5: 检索评测结果
// ================================================================
(function() {
  const s = pres.addSlide();
  addTitleBar(s, "检索与回答评测结果");
  addPageNumber(s, 5);

  // 大数字 callout
  addBigNumber(s, 0.8, 1.2, "80%", "检索 Recall@5", C.gold);
  addBigNumber(s, 2.9, 1.2, "0.817", "回答综合评分", C.teal);
  addBigNumber(s, 5.0, 1.2, "0%", "真实幻觉率", C.green);
  addBigNumber(s, 7.1, 1.2, "174", "单元测试通过", C.navy);

  // BM25 对比柱子 (用矩形模拟)
  s.addText("BM25 权重对比", { x: 0.6, y: 2.2, w: 4, h: 0.35, fontSize: 14, fontFace: "Arial", bold: true, color: C.navy, margin: 0 });

  const bm25data = [
    { label: "纯向量", val: 0.80, color: C.gold },
    { label: "BM25=0.1", val: 0.30, color: C.muted },
    { label: "BM25=0.3", val: 0.35, color: C.muted },
    { label: "BM25=0.5", val: 0.54, color: C.muted },
    { label: "BM25=0.7", val: 0.58, color: C.muted },
    { label: "BM25=0.9", val: 0.58, color: C.muted },
  ];

  bm25data.forEach((d, i) => {
    const barW = d.val * 4.0;
    const x = 1.3;
    const y = 2.65 + i * 0.42;
    s.addText(d.label, { x: 0.6, y, w: 0.7, h: 0.3, fontSize: 9, fontFace: "Arial", color: C.text, align: "right", margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y + 0.05, w: barW, h: 0.22, fill: { color: d.color } });
    s.addText((d.val * 100).toFixed(0) + "%", { x: x + barW + 0.05, y, w: 0.5, h: 0.3, fontSize: 9, fontFace: "Arial", bold: i === 0, color: i === 0 ? C.gold : C.muted, margin: 0 });
  });

  // 右侧回答评分分布
  s.addShape(pres.shapes.RECTANGLE, { x: 5.6, y: 2.2, w: 3.8, h: 2.9, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("回答质量分布 (131 条)", { x: 5.85, y: 2.35, w: 3.3, h: 0.3, fontSize: 13, fontFace: "Arial", bold: true, color: C.navy, margin: 0 });

  const dist = [
    { label: "优秀 (\u22650.8)", pct: 58.8, color: C.green },
    { label: "良好 (0.6-0.8)", pct: 29.8, color: C.teal },
    { label: "一般 (0.4-0.6)", pct: 8.4, color: C.gold },
    { label: "较差 (<0.4)", pct: 3.1, color: C.red },
  ];
  dist.forEach((d, i) => {
    const y = 2.8 + i * 0.55;
    s.addText(d.label, { x: 5.85, y, w: 1.5, h: 0.25, fontSize: 9, fontFace: "Arial", color: C.text, margin: 0 });
    const barW = d.pct * 2.0 / 100;
    s.addShape(pres.shapes.RECTANGLE, { x: 7.4, y: y + 0.02, w: barW, h: 0.2, fill: { color: d.color } });
    s.addText(d.pct.toFixed(1) + "%", { x: 7.4 + barW + 0.05, y, w: 0.6, h: 0.25, fontSize: 9, fontFace: "Arial", bold: true, color: d.color, margin: 0 });
  });

  s.addText("bge-m3 纯向量检索最优, BM25 为负优化; 88.6% 回答达良好以上, 0% 幻觉, 系统可靠可用。", {
    x: 0.6, y: 5.25, w: 8.8, h: 0.25, fontSize: 11, fontFace: "Arial", italic: true, color: C.muted,
  });
})();

// ================================================================
// Slide 6: 项目进度规划
// ================================================================
(function() {
  const s = pres.addSlide();
  addTitleBar(s, "项目进度规划");
  addPageNumber(s, 6);

  // 三周甘特图风格
  const weeks = [
    {
      week: "第 1 周", period: "阶段一",
      tasks: [
        "Ollama 安装 + qwen2.5:7b / bge-m3 部署",
        "分析北大法宝网页结构，编写爬虫",
        "爬取 30 部法律全文 (txt/JSON)",
        "文档清洗、分段、向量化",
        "构建 FAISS 索引 (3753 文档)",
      ],
      color: C.teal, status: "已完成",
    },
    {
      week: "第 2 周", period: "阶段二",
      tasks: [
        "实现基础 RAG 问答: 检索 \u2192 生成",
        "LangGraph Agent 6 节点工作流",
        "意图识别 + 查询改写 + 答案校验",
        "Vue 3 前端 + SSE 流式输出",
        "多轮对话上下文保持",
      ],
      color: C.gold, status: "已完成",
    },
    {
      week: "第 3 周", period: "阶段三",
      tasks: [
        "FastAPI 服务端 + Swagger 文档",
        "构造 131 条标注测试数据集",
        "检索评测 (6 组 BM25 对比)",
        "回答质量评测 (评分 0.817)",
        "174 单元测试 + 冒烟测试",
        "技术报告 + PPT + README",
      ],
      color: C.green, status: "已完成",
    },
  ];

  weeks.forEach((w, i) => {
    const x = 0.6 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.15, w: 2.8, h: 0.6, fill: { color: w.color }, shadow: makeShadow() });
    s.addText(w.week, { x, y: 1.15, w: 2.8, h: 0.32, fontSize: 14, fontFace: "Arial", bold: true, color: C.white, align: "center", margin: 0 });
    s.addText(w.period + "  |  " + w.status, { x, y: 1.45, w: 2.8, h: 0.25, fontSize: 9, fontFace: "Arial", color: C.white, align: "center", margin: 0, transparency: 20 });

    // 任务卡片
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.9, w: 2.8, h: 3.3, fill: { color: C.white }, shadow: makeShadow() });
    w.tasks.forEach((task, j) => {
      s.addText([{ text: task, options: { bullet: true, fontSize: 9, color: C.text } }], {
        x: x + 0.15, y: 2.05 + j * 0.5, w: 2.5, h: 0.45, paraSpaceAfter: 2, margin: 0,
      });
    });
  });

  s.addText("三周分阶段推进，每周核心任务明确，全部按期完成交付。", {
    x: 0.6, y: 5.35, w: 8.8, h: 0.2, fontSize: 11, fontFace: "Arial", italic: true, color: C.muted,
  });
})();

// ================================================================
// Slide 7: 团队分工与风险
// ================================================================
(function() {
  const s = pres.addSlide();
  addTitleBar(s, "团队分工与风险应对");
  addPageNumber(s, 7);

  // 左侧 - 分工表
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.15, w: 4.6, h: 0.4, fill: { color: C.navy } });
  s.addText("团队分工", { x: 0.6, y: 1.15, w: 4.6, h: 0.4, fontSize: 14, fontFace: "Arial", bold: true, color: C.white, align: "center", margin: 0 });

  const tableData = [
    [{ text: "角色", options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 10 } },
     { text: "负责人", options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 10 } },
     { text: "核心职责", options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 10 } }],
    [{ text: "后端", options: { fontSize: 10 } }, { text: "成员 A", options: { fontSize: 10, color: C.teal, bold: true } }, { text: "爬虫 / FastAPI / RAG 引擎", options: { fontSize: 9, color: C.muted } }],
    [{ text: "模型", options: { fontSize: 10 } }, { text: "成员 B", options: { fontSize: 10, color: C.teal, bold: true } }, { text: "Ollama 部署 / Embedding / Prompt", options: { fontSize: 9, color: C.muted } }],
    [{ text: "Agent", options: { fontSize: 10 } }, { text: "成员 C", options: { fontSize: 10, color: C.teal, bold: true } }, { text: "LangGraph / 意图识别 / 校验", options: { fontSize: 9, color: C.muted } }],
    [{ text: "前端", options: { fontSize: 10 } }, { text: "成员 D", options: { fontSize: 10, color: C.teal, bold: true } }, { text: "Vue 3 / SSE 流式 / UI 设计", options: { fontSize: 9, color: C.muted } }],
    [{ text: "测试", options: { fontSize: 10 } }, { text: "成员 E", options: { fontSize: 10, color: C.teal, bold: true } }, { text: "数据集 + 评测 + 文档 + PPT", options: { fontSize: 9, color: C.muted } }],
  ];
  s.addTable(tableData, {
    x: 0.6, y: 1.55, w: 4.6,
    border: { pt: 0.5, color: C.goldLight },
    colW: [0.9, 1.0, 2.7],
    rowH: [0.35, 0.32, 0.32, 0.32, 0.32, 0.32],
    margin: [2, 4, 2, 4],
  });

  // 右侧 - 风险矩阵
  s.addShape(pres.shapes.RECTANGLE, { x: 5.6, y: 1.15, w: 3.8, h: 0.4, fill: { color: C.red } });
  s.addText("风险与应对", { x: 5.6, y: 1.15, w: 3.8, h: 0.4, fontSize: 14, fontFace: "Arial", bold: true, color: C.white, align: "center", margin: 0 });

  const risks = [
    { risk: "CPU 推理慢 (107s)", sol: "换 GPU 或 qwen2.5:3b", color: C.red },
    { risk: "章级摘要噪声", sol: "chunk_type 过滤+Reranker", color: C.gold },
    { risk: "部分条文未命中", sol: "修正法条号映射+chunk 粒度", color: C.teal },
    { risk: "无 session 管理", sol: "引入 Redis/session_id", color: C.muted },
    { risk: "知识库静态", sol: "增量更新+版本管理", color: C.muted },
  ];
  risks.forEach((r, i) => {
    const y = 1.75 + i * 0.58;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.6, y, w: 3.8, h: 0.5, fill: { color: C.white }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: 5.6, y, w: 0.05, h: 0.5, fill: { color: r.color } });
    s.addText(r.risk, { x: 5.85, y: y + 0.02, w: 2.0, h: 0.22, fontSize: 10, fontFace: "Arial", bold: true, color: C.text, margin: 0 });
    s.addText("\u2192 " + r.sol, { x: 5.85, y: y + 0.25, w: 3.3, h: 0.2, fontSize: 9, fontFace: "Arial", color: C.muted, margin: 0 });
  });

  s.addText("5 人分工覆盖全链路, 风险预案完备; 核心风险 (推理速度/噪声) 已有解决方案或降级策略。", {
    x: 0.6, y: 5.25, w: 8.8, h: 0.25, fontSize: 11, fontFace: "Arial", italic: true, color: C.muted,
  });
})();

// ================================================================
// Slide 8: 总结
// ================================================================
(function() {
  const s = pres.addSlide();
  s.background = { color: C.dark };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });

  s.addText("总结", { x: 0.6, y: 0.25, w: 8.8, h: 0.55, fontSize: 28, fontFace: "Arial", bold: true, color: C.white, margin: 0 });

  // 关键数字
  const kpis = [
    ["80%", "检索 Recall@5"],
    ["0.817", "回答综合评分"],
    ["174", "单元测试通过"],
    ["131", "标注测试集"],
    ["30", "部法律覆盖"],
    ["0%", "真实幻觉率"],
  ];

  kpis.forEach((kpi, i) => {
    const row = Math.floor(i / 3);
    const col = i % 3;
    const x = 1.0 + col * 2.8;
    const y = 1.1 + row * 1.3;
    s.addText(kpi[0], { x, y, w: 2.4, h: 0.6, fontSize: 38, fontFace: "Arial Black", bold: true, color: C.gold, align: "center", margin: 0 });
    s.addText(kpi[1], { x, y: y + 0.62, w: 2.4, h: 0.3, fontSize: 10, fontFace: "Arial", color: C.muted, align: "center", margin: 0 });
  });

  // 核心结论
  s.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 3.8, w: 7, h: 1.2, fill: { color: C.navy }, shadow: makeShadow() });
  s.addText([
    { text: "\u2713 ", options: { color: C.gold } },
    { text: "全本地化 RAG+Agent 法律问答系统, 三周完整交付", options: { color: C.white, fontSize: 14 } },
    { text: "\n\n\u2713 ", options: { color: C.gold } },
    { text: "检索 80% + 回答 0.817 + 幻觉 0%, 质量可信", options: { color: C.white, fontSize: 14 } },
    { text: "\n\n\u2713 ", options: { color: C.gold } },
    { text: "174 测试 + Docker 一键部署 + 完整技术报告", options: { color: C.white, fontSize: 14 } },
  ], { x: 2, y: 3.9, w: 6, h: 1.0, fontSize: 14, margin: 0 });

  s.addText("Law-RAG-Agent  |  github.com/Henzhi/Law-RAG-Agent", {
    x: 2, y: 5.15, w: 6, h: 0.3, fontSize: 10, fontFace: "Arial", color: C.muted, align: "center",
  });
})();

// ================================================================
// 输出
// ================================================================
const outPath = "docs/presentation.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("PPT 已生成: " + outPath);
}).catch(err => console.error(err));
