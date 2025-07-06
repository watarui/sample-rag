"""文書ソースアダプター - Webスクレイピングの統合"""

from typing import List, Optional
from datetime import datetime

from ..domain.entities import Document
from ..domain.value_objects import DocumentContent, DocumentMetadata, ScrapingConfig, WebSource
from ..domain.services import DocumentSourceService
from ..infrastructure.web_scraping_service import WebScrapingService
from ..shared.result import Result


class DocumentSourceAdapter(DocumentSourceService):
    """文書ソースアダプター（Webスクレイピング）"""
    
    def __init__(
        self,
        web_scraping_service: WebScrapingService,
        source_type: str = "web",
    ):
        self._web_scraping_service = web_scraping_service
        self._source_type = source_type
        self._current_config: Optional[ScrapingConfig] = None
    
    async def search_documents(
        self, 
        query: str, 
        limit: int = 3
    ) -> Result[List[Document], Exception]:
        """クエリに基づいて文書を検索"""
        try:
            if self._source_type == "web":
                return await self._search_web_documents(query, limit)
            else:
                return Result.failure(Exception(f"Unsupported source type: {self._source_type}"))
        
        except Exception as e:
            return Result.failure(e)
    
    async def _search_web_documents(
        self, 
        query: str, 
        limit: int
    ) -> Result[List[Document], Exception]:
        """Webスクレイピングで文書を検索"""
        if not self._current_config:
            return Result.failure(Exception("Scraping configuration not set"))
        
        try:
            # Webスクレイピングで文書を取得
            scraping_result = await self._web_scraping_service.scrape_pages(
                config=self._current_config,
                query=query,
                limit=limit
            )
            
            if scraping_result.is_failure():
                return Result.failure(scraping_result._value)
            
            web_pages = scraping_result.unwrap()
            
            # 文書エンティティに変換
            documents = []
            for page in web_pages[:limit]:  # 念のため制限を適用
                document = Document(
                    metadata=DocumentMetadata(
                        title=page.title,
                        url=page.url,
                        created_at=page.scraped_at,
                        content_type=page.content_type,
                        keywords=[query],  # 検索クエリをキーワードとして追加
                    ),
                    content=DocumentContent(
                        text=page.content,
                        format="html",
                    ),
                    source=WebSource(
                        url=page.url,
                        scraped_at=page.scraped_at,
                    ),
                )
                documents.append(document)
            
            return Result.success(documents)
            
        except Exception as e:
            return Result.failure(Exception(f"Web scraping failed: {str(e)}"))
    
    def set_scraping_config(self, config: ScrapingConfig) -> None:
        """スクレイピング設定を設定"""
        self._current_config = config
    
    def get_scraping_config(self) -> Optional[ScrapingConfig]:
        """現在のスクレイピング設定を取得"""
        return self._current_config
    
    def get_source_type(self) -> str:
        """ソースタイプを取得"""
        return self._source_type
    
    def set_source_type(self, source_type: str) -> None:
        """ソースタイプを設定"""
        self._source_type = source_type 