"""プレゼンテーション層 - FastAPI アプリケーション"""

from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..domain.value_objects import RAGResponse, ScrapingConfig, SearchResult


# リクエスト・レスポンスモデル
class ChatRequest(BaseModel):
    query: str = Field(..., description="ユーザーの質問")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="RAGシステムの回答")
    sources: List[SearchResult] = Field(..., description="回答の根拠となった情報源")
    response_time_ms: int = Field(..., description="応答時間（ミリ秒）")


class IngestRequest(BaseModel):
    query: str = Field(..., description="検索クエリ")
    limit: int = Field(5, description="取得する文書数", ge=1, le=20)


class IngestResponse(BaseModel):
    total_chunks: int = Field(..., description="取り込んだチャンク数")
    message: str = Field(..., description="実行結果メッセージ")


class SearchRequest(BaseModel):
    query: str = Field(..., description="検索クエリ")
    limit: int = Field(5, description="取得する結果数", ge=1, le=20)


class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="検索結果")
    total_results: int = Field(..., description="総結果数")


class HealthResponse(BaseModel):
    status: str = Field(..., description="システム状態")
    services: Dict[str, bool] = Field(..., description="各サービスの状態")


class ScrapingConfigRequest(BaseModel):
    urls: List[str] = Field(..., description="スクレイピング対象URL")
    max_depth: int = Field(1, description="スクレイピング深度", ge=1, le=5)
    delay_seconds: float = Field(1.0, description="リクエスト間隔（秒）", ge=0.1, le=10.0)
    timeout_seconds: int = Field(30, description="タイムアウト（秒）", ge=5, le=120)
    max_pages: int = Field(10, description="最大ページ数", ge=1, le=100)
    respect_robots_txt: bool = Field(True, description="robots.txtを尊重")
    user_agent: str = Field("RAG-Bot/1.0", description="ユーザーエージェント")


class ScrapingTestRequest(BaseModel):
    config: ScrapingConfigRequest = Field(..., description="テスト用スクレイピング設定")
    test_query: str = Field("", description="テスト用クエリ")


class SourceConfigRequest(BaseModel):
    source_type: str = Field(pattern="^(web|hybrid)$")


def create_app(services: Dict[str, Any]) -> FastAPI:
    """FastAPIアプリケーションを作成"""
    
    app = FastAPI(
        title="RAG API",
        description="ハイブリッド型RAGシステム",
        version="1.0.0",
    )
    
    # 各ユースケースを取得
    chat_use_case = services["use_cases"]["chat_use_case"]
    search_use_case = services["use_cases"]["search_use_case"]
    document_ingestion_use_case = services["use_cases"]["document_ingestion_use_case"]
    health_use_case = services["use_cases"]["health_use_case"]
    web_scraping_config_use_case = services["use_cases"]["web_scraping_config_use_case"]
    
    @app.get("/")
    async def root():
        """ルートエンドポイント"""
        return {"message": "RAG API is running", "version": "1.0.0"}
    
    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """チャット機能"""
        result = await chat_use_case.process_query(request.query)
        
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result._value))
        
        response = result.unwrap()
        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            response_time_ms=response.response_time_ms,
        )
    
    @app.post("/api/v1/ingest", response_model=IngestResponse)
    async def ingest_documents(request: IngestRequest) -> IngestResponse:
        """文書取り込み"""
        result = await document_ingestion_use_case.ingest_documents(
            request.query, request.limit
        )
        
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result._value))
        
        total_chunks = result.unwrap()
        return IngestResponse(
            total_chunks=total_chunks,
            message=f"Successfully ingested {total_chunks} chunks",
        )
    
    @app.post("/api/v1/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        """文書検索"""
        result = await search_use_case.search(request.query, max_results=request.limit)
        
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result._value))
        
        results = result.unwrap()
        return SearchResponse(
            results=results,
            total_results=len(results),
        )
    
    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """ヘルスチェック"""
        result = await health_use_case.check_system_health()
        
        if result.is_failure():
            return HealthResponse(
                status="unhealthy",
                services={
                    "embedding_service": False,
                    "vector_search_service": False,
                    "llm_service": False,
                    "overall": False,
                },
            )
        
        health_status = result.unwrap()
        overall_status = "healthy" if health_status["overall"] else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            services=health_status,
        )
    
    # Webスクレイピング設定関連のエンドポイント
    @app.get("/api/v1/config/source")
    async def get_source_info():
        """現在の文書ソース情報を取得"""
        result = await web_scraping_config_use_case.get_source_info()
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result._value))
        return result.unwrap()
    
    @app.post("/api/v1/config/source")
    async def set_source_type(request: SourceConfigRequest):
        """文書ソースタイプを設定"""
        result = await web_scraping_config_use_case.set_source_type(request.source_type)
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result._value))
        return {"message": f"Source type set to {request.source_type}"}
    
    @app.post("/api/v1/config/scraping")
    async def set_scraping_config(request: ScrapingConfigRequest):
        """スクレイピング設定を更新"""
        config = ScrapingConfig(
            urls=request.urls,
            max_depth=request.max_depth,
            delay_seconds=request.delay_seconds,
            timeout_seconds=request.timeout_seconds,
            max_pages=request.max_pages,
            respect_robots_txt=request.respect_robots_txt,
            user_agent=request.user_agent,
        )
        
        result = await web_scraping_config_use_case.set_scraping_config(config)
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result._value))
        
        return {"message": "Scraping configuration updated successfully"}
    
    @app.post("/api/v1/config/scraping/test")
    async def test_scraping_config(request: ScrapingTestRequest):
        """スクレイピング設定をテスト"""
        config = ScrapingConfig(
            urls=request.config.urls,
            max_depth=request.config.max_depth,
            delay_seconds=request.config.delay_seconds,
            timeout_seconds=request.config.timeout_seconds,
            max_pages=request.config.max_pages,
            respect_robots_txt=request.config.respect_robots_txt,
            user_agent=request.config.user_agent,
        )
        
        result = await web_scraping_config_use_case.test_scraping_config(
            config, request.test_query
        )
        if result.is_failure():
            raise HTTPException(status_code=500, detail=str(result._value))
        
        return result.unwrap()
    
    return app 