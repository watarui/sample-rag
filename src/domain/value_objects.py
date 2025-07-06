"""ドメイン値オブジェクト"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class BaseValueObject(BaseModel):
    """基底値オブジェクト"""
    
    class Config:
        frozen = True


class QueryText(BaseValueObject):
    """クエリテキスト"""
    value: str = Field(min_length=1, max_length=1000)
    
    def __str__(self) -> str:
        return self.value


class SearchScore(BaseValueObject):
    """検索スコア"""
    value: float = Field(ge=0.0, le=1.0)
    
    def __str__(self) -> str:
        return f"{self.value:.3f}"


class Timestamp(BaseValueObject):
    """タイムスタンプ"""
    value: datetime = Field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return self.value.isoformat()


class FreshnessThreshold(BaseValueObject):
    """情報の新しさの閾値"""
    hours: int = Field(ge=1, le=24 * 7, default=24)  # 1時間〜1週間
    
    @property
    def threshold_datetime(self) -> datetime:
        return datetime.now() - timedelta(hours=self.hours)
    
    def __str__(self) -> str:
        return f"{self.hours}h"


class DocumentMetadata(BaseValueObject):
    """文書メタデータ"""
    title: str = Field(min_length=1, max_length=500)
    url: Optional[str] = None
    created_at: Optional[datetime] = None
    language: str = Field(default="ja")
    content_type: str = Field(default="text/html")
    keywords: List[str] = Field(default_factory=list)
    
    def __str__(self) -> str:
        return f"Metadata({self.title})"


class DocumentContent(BaseValueObject):
    """文書コンテンツ"""
    text: str = Field(min_length=1)
    format: str = Field(default="plain")  # plain, html, markdown
    encoding: str = Field(default="utf-8")
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    def __str__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Content({preview})"


class WebSource(BaseValueObject):
    """Web情報源"""
    url: str = Field(min_length=1)
    domain: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"Web({self.url})"


class DocumentChunk(BaseValueObject):
    """文書チャンク"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(min_length=1)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_document_id: Optional[str] = None
    chunk_index: int = Field(ge=0, default=0)
    
    @property
    def size(self) -> int:
        return len(self.content)
    
    def __str__(self) -> str:
        preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f"Chunk({preview})"


class SearchResult(BaseValueObject):
    """検索結果"""
    content: str = Field(min_length=1)
    score: SearchScore
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_id: Optional[str] = None
    
    def __str__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Result({self.score}, {preview})"


class ScrapingConfig(BaseValueObject):
    """Webスクレイピング設定"""
    urls: List[str] = Field(min_items=1)
    max_depth: int = Field(ge=1, le=5, default=1)
    delay_seconds: float = Field(ge=0.1, le=10.0, default=1.0)
    timeout_seconds: int = Field(ge=5, le=120, default=30)
    max_pages: int = Field(ge=1, le=100, default=10)
    respect_robots_txt: bool = Field(default=True)
    user_agent: str = Field(default="RAG-Bot/1.0")
    
    def __str__(self) -> str:
        return f"ScrapingConfig({len(self.urls)} URLs)"


class WebPageContent(BaseValueObject):
    """Webページコンテンツ"""
    url: str
    title: str
    content: str
    scraped_at: datetime = Field(default_factory=datetime.now)
    content_type: str = Field(default="text/html")
    
    def __str__(self) -> str:
        return f"WebPage({self.title})"


class RAGQuery(BaseModel):
    """RAGシステムへのクエリ"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: QueryText
    timestamp: Timestamp = Field(default_factory=Timestamp)
    max_results: int = Field(ge=1, le=10, default=5)
    freshness_threshold: FreshnessThreshold = Field(default_factory=FreshnessThreshold)
    search_types: List[str] = Field(default=["vector", "realtime"])

    class Config:
        frozen = True

    def __str__(self) -> str:
        return f"RAGQuery({self.text})"


class RAGResponse(BaseModel):
    """RAGシステムからの応答"""
    query_id: str
    answer: str
    sources: List[SearchResult]
    timestamp: Timestamp = Field(default_factory=Timestamp)
    response_time_ms: int = Field(ge=0)

    class Config:
        frozen = True

    def __str__(self) -> str:
        return f"RAGResponse(sources={len(self.sources)})" 