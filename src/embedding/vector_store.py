"""
FAISS 向量库构建模块。

功能：
  - 将切分后的 LangChain Document + 向量写入本地 FAISS 索引
  - 支持增量构建和全量重建
  - 持久化到磁盘，启动时可快速加载
  - 支持元数据过滤检索
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .embedder import LawEmbedder

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS 向量库"""

    def __init__(
        self,
        embedder: LawEmbedder,
        persist_dir: str | Path = 'data/vector_store',
        index_name: str = 'law_index',
    ):
        """
        Args:
            embedder: LawEmbedder 实例（已实现 LangChain Embeddings 接口）
            persist_dir: 向量库持久化根目录
            index_name: 索引名称（作为子目录名）
        """
        self.embedder = embedder
        self.persist_dir = Path(persist_dir)
        self.index_name = index_name
        self._store: Optional[FAISS] = None

    @property
    def store_dir(self) -> Path:
        return self.persist_dir / self.index_name

    @property
    def store(self) -> Optional[FAISS]:
        return self._store

    # ------------------------------------------------------------------
    # 构建 / 重建
    # ------------------------------------------------------------------

    def build_from_documents(
        self,
        documents: list[Document],
        show_progress: bool = True,
    ) -> FAISS:
        """从文档列表构建 FAISS 向量库

        LawEmbedder 本身即为 LangChain Embeddings 实例，可直接传给 FAISS。

        Args:
            documents: LangChain Document 列表
            show_progress: 是否打印进度

        Returns:
            构建好的 FAISS 实例
        """
        if not documents:
            raise ValueError('documents 不能为空')

        logger.info(f'开始向量化 {len(documents)} 个文档片段...')

        # 提取文本
        texts = [doc.page_content for doc in documents]

        # 生成向量（带进度）
        embeddings = self.embedder.embed_documents_with_progress(
            texts, show_progress=show_progress,
        )

        logger.info('正在构建 FAISS 索引...')
        metadatas = [doc.metadata for doc in documents]

        self._store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=self.embedder,  # LawEmbedder 即 Embeddings 实例
            metadatas=metadatas,
        )

        logger.info(f'FAISS 索引构建完成，共 {self._store.index.ntotal} 条向量')
        return self._store

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self) -> Path:
        """保存向量库到磁盘"""
        if self._store is None:
            raise RuntimeError('尚未构建向量库，请先调用 build_from_documents()')

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.store_dir))
        logger.info(f'向量库已保存至: {self.store_dir}')
        return self.store_dir

    def load(self) -> Optional[FAISS]:
        """从磁盘加载向量库

        Returns:
            FAISS 实例；如果索引文件不存在则返回 None
        """
        index_file = self.store_dir / 'index.faiss'
        if not index_file.exists():
            logger.warning(f'索引文件不存在: {index_file}')
            return None

        logger.info(f'正在加载向量库: {self.store_dir} ...')
        self._store = FAISS.load_local(
            str(self.store_dir),
            self.embedder,  # LawEmbedder 即 Embeddings 实例
            allow_dangerous_deserialization=True,
        )
        logger.info(f'加载完成，共 {self._store.index.ntotal} 条向量')
        return self._store

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> list[Document]:
        """语义检索

        Args:
            query: 查询文本
            k: 返回文档数
            filter_dict: 元数据过滤条件，如 {'law_name': '中华人民共和国刑法'}

        Returns:
            最相关的 k 个 Document
        """
        if self._store is None:
            raise RuntimeError('向量库未初始化，请先 build 或 load')

        return self._store.similarity_search(
            query, k=k, filter=filter_dict,
        )

    def search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """带相似度分数的语义检索

        Returns:
            list[tuple[Document, float]]: (文档, L2距离) 的列表
        """
        if self._store is None:
            raise RuntimeError('向量库未初始化，请先 build 或 load')

        return self._store.similarity_search_with_score(
            query, k=k, filter=filter_dict,
        )

    def search_by_law(
        self,
        query: str,
        law_name: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """在特定法律内检索"""
        return self.search_with_score(query, k=k, filter_dict={'law_name': law_name})

    @property
    def doc_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal
