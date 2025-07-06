"""エンベッディングサービス実装"""

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from ..domain.services import EmbeddingService
from ..shared.result import Result, try_catch_async


class SentenceTransformerEmbeddingService(EmbeddingService):
    """SentenceTransformer エンベッディングサービス実装"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {model_name} (dimension: {self._dimension})")
    
    @property
    def dimension(self) -> int:
        """エンベッディングの次元数"""
        return self._dimension
    
    async def embed_text(self, text: str) -> Result[List[float], Exception]:
        """テキストをベクター化"""
        @try_catch_async
        async def _embed() -> List[float]:
            embedding = self._model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        
        return await _embed()
    
    async def embed_texts(self, texts: List[str]) -> Result[List[List[float]], Exception]:
        """複数のテキストをベクター化"""
        @try_catch_async
        async def _embed_batch() -> List[List[float]]:
            embeddings = self._model.encode(texts, convert_to_tensor=False)
            return [embedding.tolist() for embedding in embeddings]
        
        return await _embed_batch()
    
    async def embed_text_sync(self, text: str) -> List[float]:
        """同期版（内部使用）"""
        embedding = self._model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    async def embed_texts_sync(self, texts: List[str]) -> List[List[float]]:
        """同期版（内部使用）"""
        embeddings = self._model.encode(texts, convert_to_tensor=False)
        return [embedding.tolist() for embedding in embeddings]


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI エンベッディングサービス実装"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002") -> None:
        """注意: ローカルLLMを使用する方針のため、このクラスは参考実装です"""
        self._api_key = api_key
        self._model = model
        self._dimension = 1536  # text-embedding-ada-002の次元数
        print(f"Warning: OpenAI Embedding is configured but not recommended for this project")
    
    @property
    def dimension(self) -> int:
        """エンベッディングの次元数"""
        return self._dimension
    
    async def embed_text(self, text: str) -> Result[List[float], Exception]:
        """テキストをベクター化"""
        # 実装はスキップ（ローカルLLMを推奨）
        return Result.failure(Exception("OpenAI Embedding is not implemented by design"))
    
    async def embed_texts(self, texts: List[str]) -> Result[List[List[float]], Exception]:
        """複数のテキストをベクター化"""
        # 実装はスキップ（ローカルLLMを推奨）
        return Result.failure(Exception("OpenAI Embedding is not implemented by design")) 