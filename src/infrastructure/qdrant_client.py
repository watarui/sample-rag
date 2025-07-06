"""Qdrant ベクターデータベースクライアント"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SearchParams,
    VectorParams,
)

from ..domain.services import VectorSearchService
from ..domain.value_objects import (
    DocumentChunk,
    SearchResult,
    SearchScore,
)
from ..shared.result import Result, try_catch_async


class QdrantVectorSearchService(VectorSearchService):
    """Qdrant ベクター検索サービス実装"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        vector_size: int = None,
    ) -> None:
        # 環境変数から設定を読み込み
        self._host = host or os.getenv("QDRANT_HOST", "localhost")
        self._port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self._collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "documents")
        self._vector_size = vector_size or int(os.getenv("EMBEDDING_MODEL_SIZE", "384"))
        
        print(f"Connecting to Qdrant at {self._host}:{self._port}")
        
        # バージョン互換性チェックを無効化
        self._client = QdrantClient(
            host=self._host, 
            port=self._port, 
            prefer_grpc=False, 
            check_compatibility=False
        )
        self._setup_collection()
    
    def _setup_collection(self) -> None:
        """コレクションをセットアップ"""
        try:
            # コレクションが存在するかチェック
            collections = self._client.get_collections()
            collection_exists = any(
                c.name == self._collection_name 
                for c in collections.collections
            )
            
            if not collection_exists:
                # コレクションを作成
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"Created collection: {self._collection_name}")
            
        except Exception as e:
            print(f"Error setting up collection: {e}")
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 5
    ) -> Result[List[SearchResult], Exception]:
        """類似ベクター検索"""
        @try_catch_async
        async def _search() -> List[SearchResult]:
            search_result = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
            )
            
            results = []
            for point in search_result:
                # ペイロードからDocumentChunkを復元
                payload = point.payload
                
                chunk = DocumentChunk(
                    content=payload["content"],
                    embedding=query_embedding,  # 検索用のエンベッディングを使用
                    metadata=payload.get("metadata", {}),
                    chunk_index=payload.get("chunk_index", 0),
                )
                
                search_result_obj = SearchResult(
                    document_chunk=chunk,
                    score=SearchScore(value=float(point.score)),
                    search_type="vector",
                )
                results.append(search_result_obj)
            
            return results
        
        return await _search()
    
    async def search_similar_chunks(
        self, 
        query: str,
        limit: int = 5
    ) -> Result[List[SearchResult], Exception]:
        """テキストクエリでチャンクを検索（エンベッディングは外部で生成）"""
        # このメソッドは後でエンベッディングサービスと統合される予定
        # 現在は空のリストを返す
        @try_catch_async
        async def _search() -> List[SearchResult]:
            return []
        
        return await _search()
    
    async def store_chunks(
        self, 
        chunks: List[DocumentChunk]
    ) -> Result[None, Exception]:
        """チャンクを保存"""
        @try_catch_async
        async def _store() -> None:
            points = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    continue
                
                # ペイロードを作成
                payload = {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index,
                    "source_document_id": str(chunk.source_document_id) if chunk.source_document_id else None,
                }
                
                point = PointStruct(
                    id=str(chunk.id),
                    vector=chunk.embedding,
                    payload=payload,
                )
                points.append(point)
            
            if points:
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
                print(f"Stored {len(points)} chunks to Qdrant")
        
        return await _store()
    
    async def delete_chunks(self, chunk_ids: List[str]) -> Result[None, Exception]:
        """チャンクを削除"""
        @try_catch_async
        async def _delete() -> None:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=chunk_ids,
            )
            print(f"Deleted {len(chunk_ids)} chunks from Qdrant")
        
        return await _delete()
    
    async def get_collection_info(self) -> Result[Dict[str, Any], Exception]:
        """コレクション情報を取得"""
        @try_catch_async
        async def _get_info() -> Dict[str, Any]:
            info = self._client.get_collection(self._collection_name)
            return {
                "name": self._collection_name,
                "vectors_count": getattr(info, 'vectors_count', 0),
                "points_count": getattr(info, 'points_count', 0),
                "status": str(getattr(info, 'status', 'unknown')),
            }
        
        return await _get_info()
    
    def close(self) -> None:
        """クライアントを閉じる"""
        self._client.close() 