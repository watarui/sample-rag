"""RAGサービス実装"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import time

from ..domain.entities import Document
from ..domain.services import EmbeddingService, LLMService, VectorSearchService
from ..domain.value_objects import RAGQuery, RAGResponse, SearchResult, Timestamp
from ..shared.result import Result, try_catch_async


class RAGService:
    """RAGサービス - 高度なエラーハンドリングと非同期処理を備えた本格的なRAG実装"""
    
    def __init__(
        self,
        vector_search_service: VectorSearchService,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ) -> None:
        self._vector_search_service = vector_search_service
        self._embedding_service = embedding_service
        self._llm_service = llm_service
        print("Initialized RAG service")
    
    async def process_query(self, query: str, limit: int = 5) -> Result[RAGResponse, Exception]:
        """クエリを処理してRAG回答を生成"""
        @try_catch_async
        async def _process() -> RAGResponse:
            start_time = Timestamp()
            
            # フォールバック用のLLM回答生成
            async def generate_fallback_response(error_id: str, error_msg: str) -> RAGResponse:
                print(f"{error_msg}")
                llm_result = await self._llm_service.generate_answer(query, context=[])
                
                response = llm_result.match(
                    success_func=lambda r: r,
                    failure_func=lambda e: f"エラーが発生しました: {e}"
                )
                
                end_time = Timestamp()
                return RAGResponse(
                    query_id=error_id,
                    answer=response,
                    sources=[],
                    response_time_ms=int((end_time.value - start_time.value).total_seconds() * 1000),
                )
            
            # 処理パイプライン
            embedding_result = await self._embedding_service.embed_text(query)
            
            # エンベッディング失敗時のフォールバック
            if embedding_result.is_failure():
                return await generate_fallback_response(
                    "embedding_failed", 
                    f"Embedding generation failed: {embedding_result._value}"
                )
            
            # ベクター検索と回答生成
            search_result = await (
                embedding_result
                .bind_async(lambda embedding: self._vector_search_service.search_similar(embedding, limit=limit))
                .tap_error(lambda e: print(f"Vector search failed: {e}"))
            )
            
            # 検索失敗時のフォールバック
            if search_result.is_failure():
                return await generate_fallback_response(
                    "search_failed",
                    f"Vector search failed: {search_result._value}"
                )
            
            # 成功時の処理
            final_result = await (
                search_result
                .map(lambda results: self._rank_and_filter_results(results, query))
                .tap(lambda results: print(f"Found {len(results)} relevant chunks"))
                .bind_async(lambda results: self._llm_service.generate_answer(query, context=results))
                .map(lambda answer: RAGResponse(
                    query_id="success",
                    answer=answer,
                    sources=search_result.unwrap(),
                    response_time_ms=int((Timestamp().value - start_time.value).total_seconds() * 1000),
                ))
            )
            
            return final_result.match(
                success_func=lambda response: response,
                failure_func=lambda e: RAGResponse(
                    query_id="llm_failed",
                    answer=f"回答生成中にエラーが発生しました: {e}",
                    sources=[],
                    response_time_ms=int((Timestamp().value - start_time.value).total_seconds() * 1000),
                )
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
            # 検索パイプライン
            result = await (
                (await self._embedding_service.embed_text(query))
                .tap_error(lambda e: print(f"Embedding generation failed: {e}"))
                .bind_async(lambda embedding: self._vector_search_service.search_similar(embedding, limit=limit))
                .tap_error(lambda e: print(f"Vector search failed: {e}"))
                .map(lambda results: self._rank_and_filter_results(results, query))
            )
            
            # デフォルト値として空リストを返す
            return result | []
        
        return await _search()
    
    async def process_query_with_options(
        self, 
        query: str, 
        use_fallback: bool = True,
        limit: int = 5,
        min_score: float = 0.0,
        include_metadata: bool = False
    ) -> Result[RAGResponse, Exception]:
        """オプション付きクエリ処理"""
        start_time = Timestamp()
        
        # ヘルパー関数群
        def create_response(query_id: str, answer: str, sources: List[SearchResult], metadata: Optional[Dict[str, Any]] = None) -> RAGResponse:
            response = RAGResponse(
                query_id=query_id,
                answer=answer,
                sources=sources,
                response_time_ms=int((Timestamp().value - start_time.value).total_seconds() * 1000),
            )
            
            if include_metadata and metadata:
                # 必要に応じてメタデータを追加（将来の拡張用）
                pass
            
            return response
        
        async def fallback_answer() -> Result[RAGResponse, Exception]:
            if not use_fallback:
                return Result.failure(Exception("Fallback disabled"))
            
            llm_result = await self._llm_service.generate_answer(query, context=[])
            return llm_result.map(lambda answer: create_response("fallback", answer, []))
        
        # メイン処理
        embedding_result = await self._embedding_service.embed_text(query)
        
        if embedding_result.is_failure():
            return await fallback_answer()
        
        # パイプライン処理
        search_result = await (
            embedding_result
            .bind_async(lambda embedding: self._vector_search_service.search_similar(embedding, limit=limit))
            .map(lambda results: self._filter_by_score(results, min_score))
            .map(lambda results: self._rank_and_filter_results(results, query))
        )
        
        if search_result.is_failure():
            return await fallback_answer()
        
        # LLM回答生成
        llm_result = await search_result.bind_async(
            lambda results: self._llm_service.generate_answer(query, context=results)
        )
        
        # 最終結果の生成
        metadata = {
            "search_results_count": len(search_result.unwrap()),
            "min_score_threshold": min_score,
            "processing_options": {
                "use_fallback": use_fallback,
                "limit": limit,
                "include_metadata": include_metadata,
            }
        } if include_metadata else None
        
        return llm_result.map(
            lambda answer: create_response("success", answer, search_result.unwrap(), metadata)
        ).match(
            success_func=lambda response: Result.success(response),
            failure_func=lambda e: Result.failure(e)
        )
    
    def _rank_and_filter_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """検索結果のランキングとフィルタリング"""
        if not results:
            return results
        
        # 重複除去（同じソースからの結果を統合）
        unique_results = self._remove_duplicates(results)
        
        # スコアベースのフィルタリング
        filtered_results = [r for r in unique_results if r.score.value > 0.1]
        
        # 再ランキング（必要に応じて）
        ranked_results = self._rerank_results(filtered_results, query)
        
        return ranked_results
    
    def _remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """重複する検索結果を除去"""
        seen_content = set()
        unique_results = []
        
        for result in results:
            # コンテンツの最初の100文字をハッシュキーとして使用
            content_key = result.content[:100].strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        return unique_results
    
    def _rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """検索結果の再ランキング"""
        # 簡単なキーワードマッチングベースの再ランキング
        query_words = set(query.lower().split())
        
        def calculate_relevance_score(result: SearchResult) -> float:
            content_words = set(result.content.lower().split())
            # キーワードマッチング率を計算
            keyword_match_ratio = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
            # 元のスコアとキーワードマッチング率を組み合わせ
            return result.score.value * 0.7 + keyword_match_ratio * 0.3
        
        # 再ランキング
        for result in results:
            result.score.value = calculate_relevance_score(result)
        
        # スコア順にソート
        return sorted(results, key=lambda r: r.score.value, reverse=True)
    
    def _filter_by_score(self, results: List[SearchResult], min_score: float) -> List[SearchResult]:
        """スコアによるフィルタリング"""
        return [r for r in results if r.score.value >= min_score]
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """サービスメトリクスを取得"""
        return {
            "service_name": "RAGService",
            "version": "1.0.0",
            "features": [
                "advanced_error_handling",
                "fallback_mechanism",
                "result_ranking",
                "duplicate_removal",
                "score_filtering",
                "flexible_options"
            ],
            "supported_operations": [
                "process_query",
                "search_documents", 
                "process_query_with_options"
            ]
        } 