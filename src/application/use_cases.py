"""ユースケース実装 - アプリケーション層のビジネスロジック"""

from __future__ import annotations

from typing import List

from ..domain.services import (
    DocumentProcessingService,
    DocumentSourceService,
    EmbeddingService,
    LLMService,
    VectorSearchService,
)
from ..domain.value_objects import (
    QueryText,
    RAGQuery,
    RAGResponse,
    ScrapingConfig,
    SearchResult,
)
from ..shared.result import Result


class ChatUseCase:
    """チャットユースケース"""
    
    def __init__(self, rag_service) -> None:
        self._rag_service = rag_service
    
    async def process_query(self, query_text: str) -> Result[RAGResponse, Exception]:
        """クエリを処理"""
        try:
            # RAGServiceに処理を委任
            return await self._rag_service.process_query(query_text)
            
        except Exception as e:
            return Result.failure(e)


class DocumentIngestionUseCase:
    """文書取り込みユースケース"""
    
    def __init__(
        self,
        document_processing_service: DocumentProcessingService,
        vector_search_service: VectorSearchService,
        document_source_service: DocumentSourceService,
    ) -> None:
        self._document_processing_service = document_processing_service
        self._vector_search_service = vector_search_service
        self._document_source_service = document_source_service
    
    async def ingest_documents(
        self, 
        query: str, 
        limit: int = 5
    ) -> Result[int, Exception]:
        """文書を取り込み"""
        try:
            # 1. 文書ソースから文書を取得
            print(f"Searching documents for query: {query}")
            
            documents_result = await self._document_source_service.search_documents(
                query=query,
                limit=limit
            )
            
            if documents_result.is_failure():
                return Result.failure(documents_result._value)
            
            documents = documents_result.unwrap()
            print(f"Found {len(documents)} documents")
            
            if not documents:
                return Result.success(0)
            
            # 2. 文書を処理（チャンクに分割）
            all_chunks = []
            for doc in documents:
                print(f"Processing document: {doc.metadata.title}")
                
                chunks_result = await self._document_processing_service.process_document(doc)
                if chunks_result.is_success():
                    chunks = chunks_result.unwrap()
                    all_chunks.extend(chunks)
                else:
                    print(f"Failed to process document: {chunks_result._value}")
                    continue
            
            print(f"Generated {len(all_chunks)} chunks")
            
            # 3. ベクターデータベースに保存
            if all_chunks:
                storage_result = await self._vector_search_service.store_chunks(all_chunks)
                if storage_result.is_failure():
                    return Result.failure(storage_result._value)
                
                print(f"Stored {len(all_chunks)} chunks in vector database")
            
            return Result.success(len(all_chunks))
            
        except Exception as e:
            return Result.failure(e)


class WebScrapingConfigUseCase:
    """Webスクレイピング設定ユースケース"""
    
    def __init__(self, document_source_service: DocumentSourceService) -> None:
        self._document_source_service = document_source_service
    
    async def get_source_info(self) -> Result[dict, Exception]:
        """現在の文書ソース情報を取得"""
        try:
            if hasattr(self._document_source_service, 'get_source_info'):
                info = self._document_source_service.get_source_info()
                return Result.success(info)
            else:
                return Result.success({"source_type": "unknown"})
        except Exception as e:
            return Result.failure(e)
    
    async def set_scraping_config(self, config: ScrapingConfig) -> Result[None, Exception]:
        """スクレイピング設定を更新"""
        try:
            if hasattr(self._document_source_service, 'set_scraping_config'):
                await self._document_source_service.set_scraping_config(config)
                return Result.success(None)
            else:
                return Result.failure(Exception("Web scraping not supported"))
        except Exception as e:
            return Result.failure(e)
    
    async def set_source_type(self, source_type: str) -> Result[None, Exception]:
        """文書ソースタイプを変更"""
        try:
            if hasattr(self._document_source_service, 'set_source_type'):
                self._document_source_service.set_source_type(source_type)
                return Result.success(None)
            else:
                return Result.failure(Exception("Source type switching not supported"))
        except Exception as e:
            return Result.failure(e)
    
    async def test_scraping_config(self, config: ScrapingConfig, test_query: str = "") -> Result[dict, Exception]:
        """スクレイピング設定をテスト"""
        try:
            # 一時的に設定を適用
            await self.set_scraping_config(config)
            
            # テスト実行
            test_result = await self._document_source_service.search_documents(
                test_query or "test", 
                limit=1
            )
            
            if test_result.is_success():
                documents = test_result.unwrap()
                return Result.success({
                    "success": True,
                    "documents_found": len(documents),
                    "test_document": {
                        "title": documents[0].title if documents else None,
                        "content_length": len(documents[0].content.text) if documents else 0,
                    } if documents else None,
                })
            else:
                return Result.success({
                    "success": False,
                    "error": str(test_result._value),
                })
        
        except Exception as e:
            return Result.failure(e)


class SearchUseCase:
    """検索ユースケース"""
    
    def __init__(self, rag_service) -> None:
        self._rag_service = rag_service
    
    async def search(
        self, 
        query_text: str, 
        search_types: List[str] = ["vector", "realtime"],
        max_results: int = 5,
    ) -> Result[List[SearchResult], Exception]:
        """検索を実行"""
        try:
            # RAGServiceに処理を委任
            return await self._rag_service.search_documents(query_text, max_results)
            
        except Exception as e:
            return Result.failure(e)


class SystemHealthUseCase:
    """システムヘルスチェックユースケース"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_search_service: VectorSearchService,
        llm_service: LLMService,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_search_service = vector_search_service
        self._llm_service = llm_service
    
    async def check_system_health(self) -> Result[dict, Exception]:
        """システムヘルスチェック"""
        try:
            health_status = {
                "embedding_service": False,
                "vector_search_service": False,
                "llm_service": False,
                "overall": False,
            }
            
            # エンベッディングサービスチェック
            try:
                test_result = await self._embedding_service.embed_text("test")
                health_status["embedding_service"] = test_result.is_success()
            except Exception:
                health_status["embedding_service"] = False
            
            # ベクター検索サービスチェック
            try:
                # 簡単な検索テスト
                test_embedding = [0.1] * 384  # テスト用ダミーエンベッディング
                search_result = await self._vector_search_service.search_similar(
                    test_embedding, limit=1
                )
                health_status["vector_search_service"] = search_result.is_success()
            except Exception:
                health_status["vector_search_service"] = False
            

            
            # LLMサービスチェック（軽量版）
            try:
                # 実際の生成ではなく、モデルの利用可能性をチェック
                if hasattr(self._llm_service, 'check_model_availability'):
                    availability_result = await self._llm_service.check_model_availability()
                    health_status["llm_service"] = availability_result.is_success()
                else:
                    # フォールバック：サービスが存在することを確認
                    health_status["llm_service"] = True
            except Exception:
                health_status["llm_service"] = False
            
            # 全体の状態
            health_status["overall"] = all([
                health_status["embedding_service"],
                health_status["vector_search_service"],
                health_status["llm_service"],
            ])
            
            return Result.success(health_status)
            
        except Exception as e:
            return Result.failure(e) 