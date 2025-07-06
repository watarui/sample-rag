"""エンティティ - ドメインの識別可能なオブジェクト"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .value_objects import (
    DocumentChunk,
    DocumentContent,
    DocumentId,
    Timestamp,
    WikipediaSource,
    WebSource,
)


class Document(BaseModel):
    """文書エンティティ"""
    id: DocumentId = Field(default_factory=DocumentId)
    title: str = Field(min_length=1)
    content: DocumentContent
    source: Optional[Union[WikipediaSource, WebSource]] = None
    created_at: Timestamp = Field(default_factory=Timestamp)
    updated_at: Timestamp = Field(default_factory=Timestamp)
    tags: List[str] = Field(default_factory=list)

    def create_chunks(self, chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """文書を断片に分割"""
        text = self.content.text
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        # 最小チャンクサイズを設定
        min_chunk_size = 50
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # 文章の境界で調整（効率的な実装）
            if end < len(text) and not text[end].isspace():
                # 最後の文字が区切り文字でない場合、前の区切り文字まで戻る
                search_range = min(200, end - start)  # 最大200文字まで戻って検索
                for i in range(end - 1, max(start, end - search_range), -1):
                    if text[i] in ".。!！?？\n":
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            # 最小サイズチェック
            if len(chunk_text) < min_chunk_size and chunk_index > 0:
                # 小さすぎる場合は前のチャンクに結合
                if chunks:
                    prev_chunk = chunks[-1]
                    combined_content = prev_chunk.content + " " + chunk_text
                    
                    # 結合後のチャンクを作成
                    updated_chunk = DocumentChunk(
                        id=prev_chunk.id,
                        content=combined_content,
                        metadata=prev_chunk.metadata,
                        source_document_id=prev_chunk.source_document_id,
                        chunk_index=prev_chunk.chunk_index,
                    )
                    chunks[-1] = updated_chunk
                break
            
            if chunk_text:  # 空でない場合のみ追加
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "title": self.title,
                        "source": str(self.source) if self.source else None,
                        "created_at": str(self.created_at),
                        "chunk_length": len(chunk_text),
                        **self.content.metadata,
                    },
                    source_document_id=self.id,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # 次の開始位置を計算
            start = max(end - overlap, start + 1)
            
            if end >= len(text):
                break
        
        return chunks

    def update_content(self, new_content: DocumentContent) -> Document:
        """内容を更新（イミュータブル）"""
        return Document(
            id=self.id,
            title=self.title,
            content=new_content,
            source=self.source,
            created_at=self.created_at,
            updated_at=Timestamp(),
            tags=self.tags,
        )

    def add_tag(self, tag: str) -> Document:
        """タグを追加（イミュータブル）"""
        new_tags = self.tags + [tag] if tag not in self.tags else self.tags
        return Document(
            id=self.id,
            title=self.title,
            content=self.content,
            source=self.source,
            created_at=self.created_at,
            updated_at=self.updated_at,
            tags=new_tags,
        )

    def __str__(self) -> str:
        return f"Document({self.id}): {self.title}"


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