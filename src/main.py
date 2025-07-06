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
from .infrastructure.rag_service import RAGService
from .infrastructure.web_scraping_service import WebScrapingServiceImpl

from .infrastructure.document_source_adapter import DocumentSourceAdapter
from .domain.value_objects import ScrapingConfig
from .presentation.api import create_app


async def create_services() -> Dict[str, Any]:
    """サービスを作成"""
    print("Creating services...")
    
    # 環境変数から設定を読み込み
    document_source = os.getenv("DOCUMENT_SOURCE", "web")
    portal_urls = os.getenv("PORTAL_URLS", "https://weather.yahoo.co.jp/weather/jp/13/4410.html")
    
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "documents")
    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    print(f"Document source: {document_source}")
    
    # 基底サービス
    qdrant_client = QdrantVectorSearchService(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=qdrant_collection,
    )
    
    embedding_service = SentenceTransformerEmbeddingService(
        model_name=embedding_model
    )
    
    ollama_service = OllamaLLMService(
        host=ollama_host,
        model=ollama_model,
    )
    
    # Webスクレイピングサービス
    web_scraping_service = WebScrapingServiceImpl()
    
    # 文書ソースアダプタ
    document_source_adapter = DocumentSourceAdapter(
        web_scraping_service=web_scraping_service,
        source_type=document_source,
    )
    
    # 初期設定
    if document_source == "web":
        config = ScrapingConfig(
            urls=portal_urls.split(","),
            max_depth=int(os.getenv("SCRAPING_MAX_DEPTH", "1")),
            delay_seconds=float(os.getenv("SCRAPING_DELAY_SECONDS", "1.0")),
            timeout_seconds=int(os.getenv("SCRAPING_TIMEOUT_SECONDS", "30")),
            max_pages=int(os.getenv("SCRAPING_MAX_PAGES", "5")),
            respect_robots_txt=os.getenv("SCRAPING_RESPECT_ROBOTS_TXT", "true").lower() == "true",
            user_agent=os.getenv("SCRAPING_USER_AGENT", "RAG-Bot/1.0"),
        )
        document_source_adapter.set_scraping_config(config)
        print(f"Configured web scraping for URLs: {portal_urls}")
    
    # 高次サービス
    document_processing_service = DocumentProcessingService(embedding_service)
    rag_service = RAGService(
        vector_search_service=qdrant_client,
        embedding_service=embedding_service,
        llm_service=ollama_service,
    )
    
    # ユースケース
    chat_use_case = ChatUseCase(rag_service)
    search_use_case = SearchUseCase(rag_service)
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
            "rag_service": rag_service,
        },
        "use_cases": {
            "chat_use_case": chat_use_case,
            "search_use_case": search_use_case,
            "document_ingestion_use_case": document_ingestion_use_case,
            "health_use_case": health_use_case,
            "web_scraping_config_use_case": web_scraping_config_use_case,
        },
    }


async def main():
    """メイン関数"""
    print("Starting RAG application...")
    
    # サービスを作成
    services_dict = await create_services()
    
    # FastAPIアプリケーションを作成
    app = create_app(services_dict)
    
    print("RAG application setup complete!")
    print(f"Document source: {os.getenv('DOCUMENT_SOURCE', 'web')}")
    print(f"Portal URLs: {os.getenv('PORTAL_URLS', 'Not configured')}")
    
    # サービス情報を表示
    web_scraping_config_use_case = services_dict["use_cases"]["web_scraping_config_use_case"]
    source_info_result = await web_scraping_config_use_case.get_source_info()
    if source_info_result.is_success():
        source_info = source_info_result.unwrap()
        print(f"Source info: {source_info}")
    
    # FastAPIサーバーを起動
    print("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    asyncio.run(main()) 