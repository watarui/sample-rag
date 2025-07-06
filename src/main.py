"""メインアプリケーション"""

import asyncio
import os
from typing import Dict, Any

import uvicorn

from .application.use_cases import (
    ChatUseCase,
    DocumentIngestionUseCase,
    SearchUseCase,
    SystemHealthUseCase,
    WebScrapingConfigUseCase,
)
from .domain.services import DocumentProcessingService
from .infrastructure.embedding_service import SentenceTransformerEmbeddingService
from .infrastructure.ollama_service import OllamaLLMService
from .infrastructure.qdrant_client import QdrantVectorSearchService
from .infrastructure.simple_rag_service import SimpleRAGService
from .infrastructure.web_scraping_service import WebScrapingServiceImpl

from .infrastructure.document_source_adapter import DocumentSourceAdapter
from .domain.value_objects import ScrapingConfig
from .presentation.api import Container, app, set_container


async def create_services() -> Dict[str, Any]:
    """サービスを作成"""
    print("Creating services...")
    
    # 基本サービス
    qdrant_client = QdrantVectorSearchService()
    embedding_service = SentenceTransformerEmbeddingService()
    ollama_service = OllamaLLMService()
    
    # 文書ソース（Webスクレイピング）
    web_scraping_service = WebScrapingServiceImpl()
    
    # デフォルトのスクレイピング設定
    default_scraping_config = ScrapingConfig(
        base_urls=[
            "https://weather.yahoo.co.jp/weather/jp/13/4410.html",  # Yahoo天気（東京）
        ],
        allowed_domains=["weather.yahoo.co.jp"],
        max_depth=1,
        delay_seconds=1.0,
        timeout_seconds=30,
        max_pages=5,
        respect_robots_txt=True,
        user_agent="RAG-Bot/1.0",
        content_selectors=[
            ".forecastCity",
            ".weather",
            ".forecast",
            "#main",
            ".yjw_main_md",
            ".forecastCity",
        ],
        exclude_selectors=[
            "nav",
            "footer", 
            "header",
            ".navigation",
            ".sidebar",
            ".ads",
            ".yjw_gnav",
            ".yjw_footer",
        ],
        title_selectors=[
            "h1",
            ".yjw_main_md h1",
            ".forecastCity h1",
        ],
    )
    
    # 環境変数から設定を読み込み
    portal_urls = os.getenv("PORTAL_URLS", "").split(",")
    if portal_urls and portal_urls[0].strip():
        default_scraping_config.base_urls = [url.strip() for url in portal_urls if url.strip()]
        print(f"Using custom portal URLs: {default_scraping_config.base_urls}")
    
    # 統合文書ソースアダプター
    document_source_adapter = DocumentSourceAdapter(
        web_scraping_service=web_scraping_service,
        scraping_config=default_scraping_config,
    )
    
    # 高次サービス
    document_processing_service = DocumentProcessingService(embedding_service)
    simple_rag_service = SimpleRAGService(
        vector_search_service=qdrant_client,
        embedding_service=embedding_service,
        llm_service=ollama_service,
    )
    
    # ユースケース
    chat_use_case = ChatUseCase(simple_rag_service)
    search_use_case = SearchUseCase(simple_rag_service)
    document_ingestion_use_case = DocumentIngestionUseCase(
        document_processing_service,
        qdrant_client,
        document_source_adapter,
    )
    health_use_case = SystemHealthUseCase(
        embedding_service,
        qdrant_client,
        ollama_service,
    )
    web_scraping_config_use_case = WebScrapingConfigUseCase(document_source_adapter)
    
    return {
        "services": {
            "qdrant_client": qdrant_client,
            "embedding_service": embedding_service,
            "ollama_service": ollama_service,
            "web_scraping_service": web_scraping_service,
            "document_source_adapter": document_source_adapter,
            "document_processing_service": document_processing_service,
            "simple_rag_service": simple_rag_service,
        },
        "use_cases": {
            "chat_use_case": chat_use_case,
            "search_use_case": search_use_case,
            "document_ingestion_use_case": document_ingestion_use_case,
            "health_use_case": health_use_case,
            "web_scraping_config_use_case": web_scraping_config_use_case,
        },
    }


async def setup_container() -> Container:
    """コンテナを設定"""
    print("Setting up container...")
    
    services_dict = await create_services()
    
    container = Container()
    container.chat_use_case = services_dict["use_cases"]["chat_use_case"]
    container.search_use_case = services_dict["use_cases"]["search_use_case"]
    container.ingestion_use_case = services_dict["use_cases"]["document_ingestion_use_case"]
    container.health_use_case = services_dict["use_cases"]["health_use_case"]
    container.web_scraping_config_use_case = services_dict["use_cases"]["web_scraping_config_use_case"]
    
    return container


async def initialize_application():
    """アプリケーションを初期化"""
    print("Starting RAG application...")
    
    # コンテナを設定
    container = await setup_container()
    set_container(container)
    
    print("RAG application setup complete!")
    print(f"Document source: {os.getenv('DOCUMENT_SOURCE', 'wikipedia')}")
    print(f"Portal URLs: {os.getenv('PORTAL_URLS', 'Not configured')}")
    
    # サービス情報を表示
    if container.web_scraping_config_use_case:
        source_info_result = await container.web_scraping_config_use_case.get_source_info()
        if source_info_result.is_success():
            source_info = source_info_result.unwrap()
            print(f"Source info: {source_info}")


def main():
    """メイン関数"""
    # アプリケーションを初期化
    asyncio.run(initialize_application())
    
    # FastAPIサーバーを起動
    print("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main() 