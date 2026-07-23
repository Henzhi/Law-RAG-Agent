"""
Embedding 后端抽象基类。

定义统一的向量化接口，支持 Ollama 和 OpenAI 兼容 API 两种后端。
所有后端实现必须继承此基类并实现对应方法。

同时提供 LangChain Embeddings 兼容包装器，用于 FAISS / LangChain 集成。
"""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Embedding 后端抽象基类

    子类需实现:
      - _embed_batch_impl(): 批量向量化
      - get_dimension(): 返回向量维度
      - get_model_name(): 返回模型标识
    """

    def __init__(
        self,
        model: str,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化

        Args:
            texts: 文本列表

        Returns:
            与 texts 等长的向量列表
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embs = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embs)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """单条查询文本向量化"""
        result = self._embed_batch_with_retry([text])
        return result[0]

    def embed_with_progress(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """带进度显示的批量向量化"""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embs = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embs)

            if show_progress:
                done = min(i + self.batch_size, total)
                logger.info(f'Embedding 进度: {done}/{total}')

        return all_embeddings

    # ------------------------------------------------------------------
    # 抽象方法（子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """单批向量化实现（不含重试）"""
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """返回向量维度"""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型标识，用于 pgvector embedding_model 维度隔离"""
        ...

    # ------------------------------------------------------------------
    # 重试逻辑
    # ------------------------------------------------------------------

    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """带重试的批量向量化"""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._embed_batch_impl(texts)
            except Exception as e:
                last_error = e
                logger.warning(
                    f'Embedding 调用失败 (尝试 {attempt}/{self.max_retries}): {e}'
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f'Embedding 调用失败，已重试 {self.max_retries} 次: {last_error}'
        )
