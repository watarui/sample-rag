"""FastAPI APIサーバー"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..application.use_cases import (
    ChatUseCase,
    DocumentIngestionUseCase,
    SearchUseCase,
    SystemHealthUseCase,
    WebScrapingConfigUseCase,
)
from ..shared.result import Result
from ..domain.value_objects import ScrapingConfig


# リクエスト/レスポンスモデル
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    response_time_ms: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    search_types: List[str] = Field(default=["vector", "realtime"])
    max_results: int = Field(ge=1, le=10, default=5)


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int


class IngestionRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(ge=1, le=10, default=5)


class IngestionResponse(BaseModel):
    total_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]


class ScrapingConfigRequest(BaseModel):
    """スクレイピング設定リクエスト"""
    base_urls: List[str] = Field(min_items=1)
    allowed_domains: Optional[List[str]] = None
    max_depth: int = Field(ge=0, le=10, default=2)
    delay_seconds: float = Field(ge=0, le=10, default=1.0)
    timeout_seconds: int = Field(ge=1, le=60, default=30)
    max_pages: int = Field(ge=1, le=1000, default=100)
    respect_robots_txt: bool = True
    user_agent: str = "RAG-Bot/1.0"
    content_selectors: Optional[List[str]] = None
    exclude_selectors: Optional[List[str]] = None
    title_selectors: Optional[List[str]] = None


class SourceTypeRequest(BaseModel):
    """文書ソースタイプ変更リクエスト"""
    source_type: str = Field(pattern="^(wikipedia|web|hybrid)$")


class ScrapingTestRequest(BaseModel):
    """スクレイピングテストリクエスト"""
    config: ScrapingConfigRequest
    test_query: str = ""


# 依存性注入コンテナ
class Container:
    def __init__(self) -> None:
        self.chat_use_case: ChatUseCase = None
        self.search_use_case: SearchUseCase = None
        self.ingestion_use_case: DocumentIngestionUseCase = None
        self.health_use_case: SystemHealthUseCase = None
        self.web_scraping_config_use_case: WebScrapingConfigUseCase = None


# グローバルコンテナ
container = Container()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル"""
    # 起動時の処理
    print("Starting Hybrid RAG API...")
    yield
    # 終了時の処理
    print("Shutting down Hybrid RAG API...")


# FastAPIアプリケーション
app = FastAPI(
    title="Hybrid RAG API",
    description="Wikipedia をソースとしたハイブリッド型RAGシステム",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "Hybrid RAG API is running"}


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """チャットエンドポイント"""
    if not container.chat_use_case:
        raise HTTPException(status_code=500, detail="Chat service not initialized")
    
    result = await container.chat_use_case.process_query(request.query)
    
    if result.is_failure():
        raise HTTPException(status_code=500, detail=str(result._value))
    
    response = result.unwrap()
    
    # レスポンスを変換
    sources = []
    for source in response.sources:
        sources.append({
            "content": source.document_chunk.content,
            "score": source.score.value,
            "search_type": source.search_type,
            "metadata": source.document_chunk.metadata,
        })
    
    return ChatResponse(
        answer=response.answer,
        sources=sources,
        response_time_ms=response.response_time_ms,
    )


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """検索エンドポイント"""
    if not container.search_use_case:
        raise HTTPException(status_code=500, detail="Search service not initialized")
    
    result = await container.search_use_case.search(
        request.query,
        request.search_types,
        request.max_results,
    )
    
    if result.is_failure():
        raise HTTPException(status_code=500, detail=str(result._value))
    
    search_results = result.unwrap()
    
    # レスポンスを変換
    results = []
    for search_result in search_results:
        results.append({
            "content": search_result.document_chunk.content,
            "score": search_result.score.value,
            "search_type": search_result.search_type,
            "metadata": search_result.document_chunk.metadata,
        })
    
    return SearchResponse(
        results=results,
        total_count=len(results),
    )


@app.post("/api/v1/ingest", response_model=IngestionResponse)
async def ingest(request: IngestionRequest):
    """文書取り込みエンドポイント"""
    if not container.ingestion_use_case:
        raise HTTPException(status_code=500, detail="Ingestion service not initialized")
    
    result = await container.ingestion_use_case.ingest_documents(
        request.query,
        request.limit,
    )
    
    if result.is_failure():
        raise HTTPException(status_code=500, detail=str(result._value))
    
    total_chunks = result.unwrap()
    
    return IngestionResponse(
        total_chunks=total_chunks,
        message=f"Successfully ingested {total_chunks} chunks",
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """ヘルスチェックエンドポイント"""
    if not container.health_use_case:
        raise HTTPException(status_code=500, detail="Health service not initialized")
    
    result = await container.health_use_case.check_system_health()
    
    if result.is_failure():
        raise HTTPException(status_code=500, detail=str(result._value))
    
    health_status = result.unwrap()
    
    return HealthResponse(
        status="healthy" if health_status["overall"] else "unhealthy",
        services=health_status,
    )


def set_container(new_container: Container) -> None:
    """コンテナを設定"""
    global container
    container = new_container


# Webスクレイピング管理エンドポイント
@app.get("/api/v1/source/info")
async def get_source_info() -> Dict[str, Any]:
    """現在の文書ソース情報を取得"""
    if not container.web_scraping_config_use_case:
        raise HTTPException(status_code=500, detail="Web scraping config service not initialized")
    
    result = await container.web_scraping_config_use_case.get_source_info()
    if result.is_success():
        return result.unwrap()
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get source info: {result._value}",
        )


@app.post("/api/v1/source/type")
async def set_source_type(request: SourceTypeRequest) -> Dict[str, Any]:
    """文書ソースタイプを変更"""
    if not container.web_scraping_config_use_case:
        raise HTTPException(status_code=500, detail="Web scraping config service not initialized")
    
    result = await container.web_scraping_config_use_case.set_source_type(request.source_type)
    if result.is_success():
        return {
            "success": True,
            "message": f"Source type changed to {request.source_type}",
            "source_type": request.source_type,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to change source type: {result._value}",
        )


@app.post("/api/v1/scraping/config")
async def set_scraping_config(request: ScrapingConfigRequest) -> Dict[str, Any]:
    """スクレイピング設定を更新"""
    if not container.web_scraping_config_use_case:
        raise HTTPException(status_code=500, detail="Web scraping config service not initialized")
    
    # リクエストからScrapingConfigを作成
    config = ScrapingConfig(
        base_urls=request.base_urls,
        allowed_domains=request.allowed_domains or [],
        max_depth=request.max_depth,
        delay_seconds=request.delay_seconds,
        timeout_seconds=request.timeout_seconds,
        max_pages=request.max_pages,
        respect_robots_txt=request.respect_robots_txt,
        user_agent=request.user_agent,
        content_selectors=request.content_selectors or ScrapingConfig().content_selectors,
        exclude_selectors=request.exclude_selectors or ScrapingConfig().exclude_selectors,
        title_selectors=request.title_selectors or ScrapingConfig().title_selectors,
    )
    
    result = await container.web_scraping_config_use_case.set_scraping_config(config)
    if result.is_success():
        return {
            "success": True,
            "message": f"Scraping config updated for {len(config.base_urls)} base URLs",
            "config": {
                "base_urls": config.base_urls,
                "max_depth": config.max_depth,
                "max_pages": config.max_pages,
                "delay_seconds": config.delay_seconds,
            }
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update scraping config: {result._value}",
        )


@app.post("/api/v1/scraping/test")
async def test_scraping_config(request: ScrapingTestRequest) -> Dict[str, Any]:
    """スクレイピング設定をテスト"""
    if not container.web_scraping_config_use_case:
        raise HTTPException(status_code=500, detail="Web scraping config service not initialized")
    
    # リクエストからScrapingConfigを作成
    config = ScrapingConfig(
        base_urls=request.config.base_urls,
        allowed_domains=request.config.allowed_domains or [],
        max_depth=request.config.max_depth,
        delay_seconds=request.config.delay_seconds,
        timeout_seconds=request.config.timeout_seconds,
        max_pages=request.config.max_pages,
        respect_robots_txt=request.config.respect_robots_txt,
        user_agent=request.config.user_agent,
        content_selectors=request.config.content_selectors or ScrapingConfig().content_selectors,
        exclude_selectors=request.config.exclude_selectors or ScrapingConfig().exclude_selectors,
        title_selectors=request.config.title_selectors or ScrapingConfig().title_selectors,
    )
    
    result = await container.web_scraping_config_use_case.test_scraping_config(config, request.test_query)
    if result.is_success():
        return result.unwrap()
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test scraping config: {result._value}",
        ) 