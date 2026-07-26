"""
文档解析管道主流程。

将上传的 PDF/DOCX 文件解析 → 清洗 → 分块 → 向量化 → 写入 pgvector。
支持异步任务状态追踪。
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from src.knowledge.ingestion.pdf_parser import PDFParser
from src.knowledge.ingestion.docx_parser import DocxParser
from src.knowledge.ingestion.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)

# 匹配条号，如「第二百三十二条」「第232条」
_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千零两0-9]+条")


def _extract_article_range(content: str) -> str:
    """从条文内容中提取首个条号，用于检索结果的「引用条文」展示。"""
    m = _ARTICLE_RE.search(content or "")
    return m.group(0) if m else ""

# 支持的文件类型
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_MB = 50

# 任务状态
class TaskStatus:
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"


class IngestionPipeline:
    """文档解析管道

    用法:
        pipeline = IngestionPipeline(store, embedder)
        task_id = pipeline.submit("path/to/law.pdf", doc_type="law")
        status = pipeline.get_status(task_id)
    """

    def __init__(self, store, embedder):
        self._store = store          # PgvectorStore
        self._embedder = embedder    # EmbeddingAdapter
        self._pdf_parser = PDFParser()
        self._docx_parser = DocxParser()
        self._cleaner = TextCleaner()
        self._tasks: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # 任务管理
    # ------------------------------------------------------------------

    def submit(
        self,
        file_path: str,
        doc_type: str = "law",
        source: str = "",
        effective_date: str | None = None,
    ) -> str:
        """提交解析任务，返回 task_id"""
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "status": TaskStatus.PENDING,
            "file_path": file_path,
            "doc_type": doc_type,
            "source": source,
            "effective_date": effective_date,
            "progress": 0,
            "error": None,
        }
        logger.info(f"解析任务已提交: task_id={task_id[:8]}..., file={file_path}")
        return task_id

    def get_status(self, task_id: str) -> dict | None:
        """查询任务状态"""
        return self._tasks.get(task_id)

    def run(self, task_id: str) -> int:
        """同步执行解析任务，返回写入的块数量"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"任务不存在: {task_id}")

        try:
            file_path = task["file_path"]
            file_name = Path(file_path).name
            ext = Path(file_path).suffix.lower()

            # 1. 解析
            task["status"] = TaskStatus.PARSING
            raw_text = self._parse_file(file_path, ext)

            # 2. 清洗
            task["status"] = TaskStatus.CHUNKING
            cleaned = self._cleaner.clean(raw_text)
            task["progress"] = 30

            if not cleaned or len(cleaned) < 20:
                raise ValueError(f"解析后文本过短（{len(cleaned)}字符），可能为空白或扫描件")

            # 3. 创建文档记录
            doc_id = self._store.ensure_document(
                doc_type=task["doc_type"],
                title=task.get("title") or file_name.replace(ext, ""),
                source=task.get("source", ""),
                effective_date=task.get("effective_date"),
            )

            # 4. 分块 — 以段落为边界，500 字一段
            task["status"] = TaskStatus.EMBEDDING
            chunks = self._split_paragraphs(
                cleaned, doc_id, doc_type=task["doc_type"],
                title=task.get("title") or file_name.replace(ext, ""),
            )
            task["progress"] = 60

            # 5. 向量化 + 写入
            task["status"] = TaskStatus.INDEXING
            for i in range(0, len(chunks), self._embedder.batch_size):
                batch = chunks[i:i + self._embedder.batch_size]
                texts = [c["content"] for c in batch]
                embeddings = self._embedder.embed_documents(texts)
                for c, emb in zip(batch, embeddings):
                    c["embedding"] = emb
                self._store.insert_chunks(batch, embedding_model=self._embedder.model)
                task["progress"] = 60 + int(40 * (i + len(batch)) / len(chunks))

            task["status"] = TaskStatus.DONE
            task["progress"] = 100
            self._store.reindex()
            logger.info(f"解析完成: {file_name} → {len(chunks)} 块")
            return len(chunks)

        except Exception as e:
            task["status"] = TaskStatus.FAILED
            task["error"] = str(e)
            logger.error(f"解析失败: {task['file_path']} — {e}")
            raise

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def validate(file_path: str) -> tuple[bool, str]:
        """上传前校验文件合法性"""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"不支持的文件格式: {ext}，支持: {', '.join(ALLOWED_EXTENSIONS)}"
        if not path.exists():
            return False, "文件不存在"
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return False, f"文件过大: {size_mb:.1f}MB（限制 {MAX_FILE_SIZE_MB}MB）"
        return True, ""

    def _parse_file(self, file_path: str, ext: str) -> str:
        if ext == ".pdf":
            return self._pdf_parser.parse(file_path)
        elif ext == ".docx":
            return self._docx_parser.parse(file_path)
        elif ext == ".txt":
            # 中文法律文档常见 GBK/GB2312 编码，先试 UTF-8，失败回退 GBK
            try:
                return Path(file_path).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return Path(file_path).read_text(encoding="gbk")
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

    def ingest_text(
        self,
        title: str,
        text: str,
        doc_type: str = "law",
        source: str = "",
        effective_date: str | None = None,
        force: bool = False,
    ) -> int:
        """直接入库纯文本（无需落盘文件）。

        与 run() 走同一条「清洗 → 分块 → embedding → insert_chunks」链路。

        Args:
            title          : 文档标题（同时用于增量去重，按标题精确匹配）
            text           : 已清洗 / 未清洗的正文
            doc_type       : 文档类型
            source         : 来源标识（如 flk.npc.gov.cn）
            effective_date : 生效日期（ISO 字符串或 None）
            force          : 已存在时是否删除旧文档后重建

        Returns:
            0  -> 已存在且非强制（跳过）
            >0 -> 写入的文本块数量
        """
        if not text or len(text) < 20:
            raise ValueError(f"文本过短（{len(text)} 字符）")
        cleaned = self._cleaner.clean(text)
        if not cleaned or len(cleaned) < 20:
            raise ValueError("清洗后文本过短")

        existing = self._store.get_document_id_by_title(title)
        if existing and not force:
            logger.info(f"[ingest] 跳过(已存在): {title}")
            return 0
        if existing and force:
            logger.info(f"[ingest] 强制重建，删除旧文档: {title}")
            self._store.delete_document(existing)

        doc_id = self._store.ensure_document(
            doc_type=doc_type, title=title, source=source, effective_date=effective_date,
        )
        chunks = self._split_paragraphs(cleaned, doc_id, doc_type=doc_type, title=title)
        if not chunks:
            raise ValueError("分块结果为空")

        for i in range(0, len(chunks), self._embedder.batch_size):
            batch = chunks[i : i + self._embedder.batch_size]
            embeddings = self._embedder.embed_documents([c["content"] for c in batch])
            for c, emb in zip(batch, embeddings):
                c["embedding"] = emb
            self._store.insert_chunks(batch, embedding_model=self._embedder.model)
        logger.info(f"[ingest] 写入完成: {title} → {len(chunks)} 块")
        return len(chunks)

    @staticmethod
    def _split_paragraphs(
        text: str,
        doc_id: str,
        doc_type: str = "law",
        max_chars: int = 500,
        title: str | None = None,
    ) -> list[dict]:
        """按段落切分文本为块

        法律文档以条文为天然段落边界，
        每个「第X条」作为独立块，超长条文再按句号拆分。
        每个块会带上 law_name（文档标题）与 article_range（解析出的条号），
        供检索结果的「引用条文」展示使用。
        """
        chunks: list[dict] = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            if len(para) <= max_chars:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_type": "article" if doc_type == "law" else "summary",
                    "content": para,
                    "metadata": {"raw": para, "doc_type": doc_type},
                })
            else:
                # 超长段落按句号拆分
                sentences = para.split("。")
                buf = ""
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue  # 跳过句号产生的空串（如段落以"。"结尾）
                    s += "。"
                    if len(buf) + len(s) > max_chars and buf:
                        chunks.append({
                            "doc_id": doc_id,
                            "chunk_type": "article",
                            "content": buf.strip(),
                            "metadata": {"raw": para[:200], "doc_type": doc_type},
                        })
                        buf = s
                    else:
                        buf += s
                if buf.strip():
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_type": "article",
                        "content": buf.strip(),
                        "metadata": {"raw": para[:200], "doc_type": doc_type},
                    })

        # 补充法律引用字段到 metadata，供检索结果「引用条文」展示
        # （检索器从 document_chunks.metadata JSONB 读取 law_name / article_range）
        law_name = (title or "").strip()
        for i, c in enumerate(chunks):
            meta = c.setdefault("metadata", {})
            meta["law_name"] = law_name
            meta["article_range"] = _extract_article_range(c["content"])
            meta["paragraph_index"] = i
        return chunks
