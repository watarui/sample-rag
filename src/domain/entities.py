"""エンティティ - ドメインの識別可能なオブジェクト"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from .value_objects import (
    DocumentChunk,
    DocumentContent,
    DocumentMetadata,
    SearchResult,
    Timestamp,
    WebSource,
)


class Document(BaseModel):
    """文書エンティティ"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: DocumentMetadata
    content: DocumentContent
    chunks: List[DocumentChunk] = Field(default_factory=list)
    source: Optional[WebSource] = None
    created_at: Timestamp = Field(default_factory=Timestamp)
    updated_at: Optional[Timestamp] = None
    
    def add_chunk(self, chunk: DocumentChunk) -> None:
        """チャンクを追加"""
        self.chunks.append(chunk)
    
    def get_chunks_by_size(self, min_size: int = 100) -> List[DocumentChunk]:
        """指定サイズ以上のチャンクを取得"""
        return [chunk for chunk in self.chunks if chunk.size >= min_size]
    
    def get_total_content_size(self) -> int:
        """全コンテンツサイズを取得"""
        return sum(chunk.size for chunk in self.chunks)
    
    def __str__(self) -> str:
        return f"Document({self.metadata.title})"


class QueryHistory(BaseModel):
    """クエリ履歴エンティティ"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    results: List[SearchResult]
    timestamp: Timestamp = Field(default_factory=Timestamp)
    response_time_ms: int = Field(ge=0)
    
    def __str__(self) -> str:
        return f"QueryHistory({self.query[:50]}...)"


class VectorIndex(BaseModel):
    """ベクターインデックスエンティティ"""
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    distance_metric: str = Field(default="cosine")
    created_at: Timestamp = Field(default_factory=Timestamp)
    document_count: int = Field(ge=0, default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"VectorIndex({self.name}): {self.document_count} documents"


class SearchSession(BaseModel):
    """検索セッションエンティティ"""
    id: str = Field(min_length=1)
    user_id: Optional[str] = None
    started_at: Timestamp = Field(default_factory=Timestamp)
    last_activity: Timestamp = Field(default_factory=Timestamp)
    query_count: int = Field(ge=0, default=0)
    context: Dict[str, Any] = Field(default_factory=dict)

    def add_query(self) -> SearchSession:
        """クエリを追加（イミュータブル）"""
        return SearchSession(
            id=self.id,
            user_id=self.user_id,
            started_at=self.started_at,
            last_activity=Timestamp(),
            query_count=self.query_count + 1,
            context=self.context,
        )

    def update_context(self, new_context: Dict[str, Any]) -> SearchSession:
        """コンテキストを更新（イミュータブル）"""
        return SearchSession(
            id=self.id,
            user_id=self.user_id,
            started_at=self.started_at,
            last_activity=Timestamp(),
            query_count=self.query_count,
            context={**self.context, **new_context},
        )

    def __str__(self) -> str:
        return f"SearchSession({self.id}): {self.query_count} queries" 