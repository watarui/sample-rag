"""値オブジェクト - 不変なドメインの値"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class DocumentId(BaseModel):
    """文書ID"""
    value: UUID = Field(default_factory=uuid4)

    class Config:
        frozen = True

    def __str__(self) -> str:
        return str(self.value)


class QueryText(BaseModel):
    """クエリテキスト"""
    value: str = Field(min_length=1, max_length=1000)

    class Config:
        frozen = True

    @validator("value")
    def validate_value(cls, v: str) -> str:
        return v.strip()

    def __str__(self) -> str:
        return self.value


class DocumentContent(BaseModel):
    """文書内容"""
    text: str = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True

    def __str__(self) -> str:
        return self.text[:100] + "..." if len(self.text) > 100 else self.text


class DocumentChunk(BaseModel):
    """文書の断片"""
    id: DocumentId = Field(default_factory=DocumentId)
    content: str = Field(min_length=1)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_document_id: Optional[DocumentId] = None
    chunk_index: int = Field(ge=0)

    class Config:
        frozen = True

    def __str__(self) -> str:
        return f"Chunk({self.id}): {self.content[:50]}..."


class SearchScore(BaseModel):
    """検索スコア"""
    value: float = Field(ge=0.0, le=1.0)

    class Config:
        frozen = True

    def __str__(self) -> str:
        return f"{self.value:.3f}"


class SearchResult(BaseModel):
    """検索結果"""
    document_chunk: DocumentChunk
    score: SearchScore
    search_type: str = Field(default="vector")  # vector, hybrid, realtime

    class Config:
        frozen = True

    def __str__(self) -> str:
        return f"SearchResult(score={self.score}, type={self.search_type})"


class Timestamp(BaseModel):
    """タイムスタンプ"""
    value: datetime = Field(default_factory=datetime.now)

    class Config:
        frozen = True

    def __str__(self) -> str:
        return self.value.isoformat()


class FreshnessThreshold(BaseModel):
    """データの鮮度閾値（秒）"""
    value: int = Field(ge=0, default=3600)  # デフォルト1時間

    class Config:
        frozen = True

    def is_fresh(self, timestamp: Timestamp) -> bool:
        """データが新鮮かどうかを判定"""
        now = datetime.now()
        diff = (now - timestamp.value).total_seconds()
        return diff <= self.value


class WikipediaSource(BaseModel):
    """Wikipedia情報源"""
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    language: str = Field(min_length=1)

    def __str__(self) -> str:
        return f"Wikipedia({self.title})"


class WebSource(BaseModel):
    """Web情報源"""
    url: str = Field(min_length=1)
    title: Optional[str] = None
    domain: str = Field(min_length=1)
    scraped_at: Optional[Timestamp] = None
    content_type: str = Field(default="text/html")
    
    def __str__(self) -> str:
        return f"Web({self.domain}): {self.title or self.url}"


class ScrapingConfig(BaseModel):
    """スクレイピング設定"""
    base_urls: List[str] = Field(min_items=1)
    allowed_domains: List[str] = Field(default_factory=list)
    max_depth: int = Field(ge=0, le=10, default=2)
    delay_seconds: float = Field(ge=0, le=10, default=1.0)
    timeout_seconds: int = Field(ge=1, le=60, default=30)
    max_pages: int = Field(ge=1, le=1000, default=100)
    respect_robots_txt: bool = Field(default=True)
    user_agent: str = Field(default="RAG-Bot/1.0")
    
    # CSS セレクター設定
    content_selectors: List[str] = Field(default_factory=lambda: [
        "main", "article", ".content", "#content", ".main-content"
    ])
    exclude_selectors: List[str] = Field(default_factory=lambda: [
        "nav", "footer", "header", ".navigation", ".sidebar", ".ads"
    ])
    title_selectors: List[str] = Field(default_factory=lambda: [
        "h1", "title", ".page-title", ".title"
    ])


class WebPageContent(BaseModel):
    """Webページコンテンツ"""
    url: str = Field(min_length=1)
    title: str = Field(default="")
    text_content: str = Field(default="")
    html_content: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    links: List[str] = Field(default_factory=list)
    scraped_at: Timestamp = Field(default_factory=Timestamp)
    
    def get_clean_text(self) -> str:
        """クリーンなテキストを取得"""
        import re
        
        text = self.text_content
        # 余分な空白を削除
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text


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