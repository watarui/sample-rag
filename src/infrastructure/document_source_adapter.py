"""文書ソースアダプター - WikipediaとWebスクレイピングの統合"""

from __future__ import annotations

import os
from typing import List, Optional

from ..domain.entities import Document
from ..domain.services import DocumentSourceService, WebScrapingService
from ..domain.value_objects import ScrapingConfig
from ..shared.result import Result, try_catch_async


class DocumentSourceAdapter(DocumentSourceService):
    """文書ソースアダプター（WikipediaまたはWebスクレイピング）"""
    
    def __init__(
        self,
        web_scraping_service: Optional[WebScrapingService] = None,
        scraping_config: Optional[ScrapingConfig] = None,
    ) -> None:
        self._web_scraping_service = web_scraping_service
        self._scraping_config = scraping_config
        self._source_type = os.getenv("DOCUMENT_SOURCE", "web").lower()
        
        print(f"Initialized DocumentSourceAdapter with source: {self._source_type}")
    
    async def search_documents(
        self, 
        query: str, 
        limit: int = 3
    ) -> Result[List[Document], Exception]:
        """文書を検索（Webスクレイピング）"""
        @try_catch_async
        async def _search() -> List[Document]:
            if self._web_scraping_service and self._scraping_config:
                print(f"Using web scraping for query: {query}")
                result = await self._web_scraping_service.scrape_pages(
                    config=self._scraping_config,
                    query=query,
                    limit=limit
                )
                
                if result.is_success():
                    return result.unwrap()
                else:
                    print(f"Web scraping failed: {result._value}")
                    return []
            else:
                raise Exception("No web scraping service configured")
        
        return await _search()
    

    
    def get_source_info(self) -> dict:
        """現在の設定情報を取得"""
        return {
            "source_type": self._source_type,
            "web_scraping_available": self._web_scraping_service is not None,
            "scraping_config": {
                "base_urls": self._scraping_config.base_urls if self._scraping_config else [],
                "max_depth": self._scraping_config.max_depth if self._scraping_config else 0,
                "max_pages": self._scraping_config.max_pages if self._scraping_config else 0,
            } if self._scraping_config else None,
        }
    
    async def set_scraping_config(self, config: ScrapingConfig) -> None:
        """スクレイピング設定を更新"""
        self._scraping_config = config
        print(f"Updated scraping config: {len(config.base_urls)} base URLs")
    
    def set_source_type(self, source_type: str) -> None:
        """文書ソースタイプを変更"""
        if source_type.lower() in ["web"]:
            self._source_type = source_type.lower()
            print(f"Changed document source to: {self._source_type}")
        else:
            raise ValueError(f"Invalid source type: {source_type}. Only 'web' is supported.")
    
    async def close(self) -> None:
        """リソースをクリーンアップ"""
        if self._web_scraping_service and hasattr(self._web_scraping_service, 'close'):
            await self._web_scraping_service.close() 