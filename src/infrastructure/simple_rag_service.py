"""シンプルなRAGサービス実装"""

from __future__ import annotations

from typing import List

from ..domain.entities import Document
from ..domain.services import EmbeddingService, LLMService, VectorSearchService
from ..domain.value_objects import RAGQuery, RAGResponse, SearchResult, Timestamp
from ..shared.result import Result, try_catch_async


class SimpleRAGService:
    """シンプルなRAGサービス"""
    
    def __init__(
        self,
        vector_search_service: VectorSearchService,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ) -> None:
        self._vector_search_service = vector_search_service
        self._embedding_service = embedding_service
        self._llm_service = llm_service
        print("Initialized Simple RAG service")
    
    async def process_query(self, query: str) -> Result[RAGResponse, Exception]:
        """クエリを処理してRAG回答を生成"""
        @try_catch_async
        async def _process() -> RAGResponse:
            start_time = Timestamp()
            
            # 1. クエリをエンベッディングに変換
            embedding_result = await self._embedding_service.embed_text(query)
            
            if embedding_result.is_failure():
                # エンベッディング生成失敗時は直接LLMに質問
                print(f"Embedding generation failed: {embedding_result._value}")
                llm_result = await self._llm_service.generate_answer(
                    query, context=[]
                )
                
                if llm_result.is_failure():
                    raise Exception(f"LLM generation failed: {llm_result._value}")
                
                response = llm_result.unwrap()
                end_time = Timestamp()
                
                return RAGResponse(
                    query_id="embedding_failed",
                    answer=response,
                    sources=[],
                    response_time_ms=int((end_time.value - start_time.value).total_seconds() * 1000),
                )
            
            query_embedding = embedding_result.unwrap()
            
            # 2. ベクター検索で関連文書を取得
            search_result = await self._vector_search_service.search_similar(
                query_embedding, limit=5
            )
            
            if search_result.is_failure():
                # 検索失敗時は直接LLMに質問
                print(f"Vector search failed: {search_result._value}")
                llm_result = await self._llm_service.generate_answer(
                    query, context=[]
                )
                
                if llm_result.is_failure():
                    raise Exception(f"LLM generation failed: {llm_result._value}")
                
                response = llm_result.unwrap()
                end_time = Timestamp()
                
                return RAGResponse(
                    query_id="search_failed",
                    answer=response,
                    sources=[],
                    response_time_ms=int((end_time.value - start_time.value).total_seconds() * 1000),
                )
            
            # 3. 検索結果を取得
            search_results = search_result.unwrap()
            
            print(f"Found {len(search_results)} relevant chunks")
            
            # 4. LLMで回答生成
            llm_result = await self._llm_service.generate_answer(
                query, context=search_results
            )
            
            if llm_result.is_failure():
                raise Exception(f"LLM generation failed: {llm_result._value}")
            
            response = llm_result.unwrap()
            end_time = Timestamp()
            
            return RAGResponse(
                query_id="success",
                answer=response,
                sources=search_results,
                response_time_ms=int((end_time.value - start_time.value).total_seconds() * 1000),
            )
        
        return await _process()
    
    async def search_documents(
        self, 
        query: str, 
        limit: int = 5
    ) -> Result[List[SearchResult], Exception]:
        """文書検索"""
        @try_catch_async
        async def _search() -> List[SearchResult]:
            # 1. クエリをエンベッディングに変換
            embedding_result = await self._embedding_service.embed_text(query)
            
            if embedding_result.is_failure():
                print(f"Embedding generation failed: {embedding_result._value}")
                return []
            
            query_embedding = embedding_result.unwrap()
            
            # 2. ベクター検索
            search_result = await self._vector_search_service.search_similar(
                query_embedding, limit=limit
            )
            
            if search_result.is_failure():
                print(f"Vector search failed: {search_result._value}")
                return []
            
            return search_result.unwrap()
        
        return await _search() 