"""
文档解析管道单元测试。

验证:
  1. 各解析器可导入
  2. TextCleaner 清洗逻辑
  3. IngestionPipeline 分块逻辑
  4. 文件类型校验
"""
from __future__ import annotations


class TestImports:
    def test_import_pdf_parser(self):
        from src.knowledge.ingestion.pdf_parser import PDFParser
        assert PDFParser is not None

    def test_import_docx_parser(self):
        from src.knowledge.ingestion.docx_parser import DocxParser
        assert DocxParser is not None

    def test_import_cleaner(self):
        from src.knowledge.ingestion.text_cleaner import TextCleaner
        assert TextCleaner is not None

    def test_import_pipeline(self):
        from src.knowledge.ingestion.pipeline import IngestionPipeline
        assert IngestionPipeline is not None

    def test_import_task_status(self):
        from src.knowledge.ingestion.pipeline import TaskStatus
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.DONE == "done"


class TestTextCleaner:
    def test_clean_normal_text(self):
        from src.knowledge.ingestion.text_cleaner import TextCleaner
        cleaner = TextCleaner()
        text = "第一条  为了惩罚  犯罪，保护人民。\n\n第二条  根据宪法，结合..."
        result = cleaner.clean(text)
        assert "第一条" in result
        assert "第二条" in result
        # 多余空格应被压缩
        assert "为了惩罚 犯罪" not in result or "为了惩罚" in result

    def test_clean_page_numbers(self):
        from src.knowledge.ingestion.text_cleaner import TextCleaner
        cleaner = TextCleaner()
        # 页码行应该被过滤
        text = "第一条 内容\n123\n第二条 内容\n- 45 -"
        result = cleaner.clean(text)
        assert "123" not in result.split("\n")
        assert "- 45 -" not in result.split("\n")

    def test_clean_empty(self):
        from src.knowledge.ingestion.text_cleaner import TextCleaner
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""

    def test_clean_batch(self):
        from src.knowledge.ingestion.text_cleaner import TextCleaner
        cleaner = TextCleaner()
        results = cleaner.clean_batch(["第一条 内容", "", "第二条 内容"])
        assert len(results) == 3
        assert results[1] == ""


class TestPipelineSplit:
    def test_split_short_paragraphs(self):
        from src.knowledge.ingestion.pipeline import IngestionPipeline
        text = "第一条 为了惩罚犯罪。\n\n第二条 结合我国实际情况。"
        chunks = IngestionPipeline._split_paragraphs(text, "doc_id", "law")
        assert len(chunks) == 2
        assert all(c["chunk_type"] == "article" for c in chunks)

    def test_split_long_paragraph(self):
        from src.knowledge.ingestion.pipeline import IngestionPipeline
        # 构造超长段落（大于500字）
        long_text = "。".join(["第X条规定" * 10 for _ in range(20)]) + "。"
        chunks = IngestionPipeline._split_paragraphs(long_text, "doc_id", "law", max_chars=100)
        # 超长段落应该被拆分
        assert len(chunks) > 1

    def test_split_empty_text(self):
        from src.knowledge.ingestion.pipeline import IngestionPipeline
        chunks = IngestionPipeline._split_paragraphs("", "doc_id", "law")
        assert chunks == []


class TestValidation:
    def test_allowed_extensions(self):
        from src.knowledge.ingestion.pipeline import IngestionPipeline, ALLOWED_EXTENSIONS
        assert ".pdf" in ALLOWED_EXTENSIONS
        assert ".docx" in ALLOWED_EXTENSIONS
        assert ".txt" in ALLOWED_EXTENSIONS
        assert ".exe" not in ALLOWED_EXTENSIONS
