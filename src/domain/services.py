"""ドメインサービス - 複数のエンティティにまたがるビジネスロジック"""

from __future__ import annotations

from datetime import datetime
from typing import List, Protocol

from ..shared.result import Result
from .entities import Document
from .value_objects import (
    DocumentChunk,
    FreshnessThreshold,
    QueryText,
    RAGQuery,
    RAGResponse,
    SearchResult,
    SearchScore,
    Timestamp,
    ScrapingConfig,
    WebPageContent,
)


class EmbeddingService(Protocol):
    """エンベッディングサービス（インターフェース）"""
    
    async def embed_text(self, text: str) -> Result[List[float], Exception]:
        """テキストをベクター化"""
        ...
    
    async def embed_texts(self, texts: List[str]) -> Result[List[List[float]], Exception]:
        """複数のテキストをベクター化"""
        ...


class VectorSearchService(Protocol):
    """ベクター検索サービス（インターフェース）"""
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 5
    ) -> Result[List[SearchResult], Exception]:
        """類似ベクター検索"""
        ...
    
    async def store_chunks(
        self, 
        chunks: List[DocumentChunk]
    ) -> Result[None, Exception]:
        """チャンクを保存"""
        ...



class WebScrapingService(Protocol):
    """Webスクレイピングサービス（インターフェース）"""
    
    async def scrape_pages(
        self,
        config: ScrapingConfig,
        query: str = "",
        limit: int = 10
    ) -> Result[List[Document], Exception]:
        """Webページをスクレイピングして文書を生成"""
        ...
    
    async def scrape_single_page(
        self,
        url: str,
        config: ScrapingConfig
    ) -> Result[Document, Exception]:
        """単一のWebページをスクレイピング"""
        ...
    
    async def discover_pages(
        self,
        base_url: str,
        config: ScrapingConfig
    ) -> Result[List[str], Exception]:
        """ベースURLから関連ページを発見"""
        ...


class DocumentSourceService(Protocol):
    """文書ソースサービス（抽象インターフェース）"""
    
    async def search_documents(
        self, 
        query: str, 
        limit: int = 3
    ) -> Result[List[Document], Exception]:
        """文書を検索（WikipediaまたはWebスクレイピング）"""
        ...


class LLMService(Protocol):
    """LLMサービス（インターフェース）"""
    
    async def generate_answer(
        self, 
        query: str, 
        context: List[SearchResult]
    ) -> Result[str, Exception]:
        """コンテキストに基づいて回答を生成"""
        ...


class DocumentProcessingService:
    """文書処理サービス"""
    
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embedding_service = embedding_service
    
    async def process_document(
        self, 
        document: Document, 
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> Result[List[DocumentChunk], Exception]:
        """文書を処理してチャンクを生成"""
        try:
            print(f"Processing document: {document.title}")
            
            # チャンクに分割
            chunks = document.create_chunks(chunk_size, overlap)
            
            if not chunks:
                print(f"No chunks created for document: {document.title}")
                return Result.success([])
            
            print(f"Created {len(chunks)} chunks from document: {document.title}")
            
            # 各チャンクのエンベッディングを生成（バッチ処理）
            texts = [chunk.content for chunk in chunks if chunk.content.strip()]
            
            if not texts:
                print(f"No valid text content found in document: {document.title}")
                return Result.success([])
            
            print(f"Generating embeddings for {len(texts)} chunks...")
            
            # バッチサイズを制限してメモリ使用量を管理
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                embeddings_result = await self._embedding_service.embed_texts(batch_texts)
                
                if embeddings_result.is_failure():
                    print(f"Error generating embeddings for batch {i//batch_size + 1}: {embeddings_result._value}")
                    return Result.failure(embeddings_result._value)
                
                batch_embeddings = embeddings_result.unwrap()
                all_embeddings.extend(batch_embeddings)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # エンベッディングをチャンクに追加
            processed_chunks = []
            for chunk, embedding in zip(chunks, all_embeddings):
                if chunk.content.strip():  # 空のチャンクをスキップ
                    processed_chunk = DocumentChunk(
                        id=chunk.id,
                        content=chunk.content,
                        embedding=embedding,
                        metadata={
                            **chunk.metadata,
                            "embedding_dimension": len(embedding),
                            "processing_timestamp": str(datetime.now()),
                        },
                        source_document_id=chunk.source_document_id,
                        chunk_index=chunk.chunk_index,
                    )
                    processed_chunks.append(processed_chunk)
            
            print(f"Successfully processed {len(processed_chunks)} chunks for document: {document.title}")
            return Result.success(processed_chunks)
            
        except Exception as e:
            print(f"Error processing document {document.title}: {e}")
            return Result.failure(e)
    
    async def process_documents(
        self, 
        documents: List[Document], 
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> Result[List[DocumentChunk], Exception]:
        """複数の文書を並列処理"""
        try:
            print(f"Processing {len(documents)} documents...")
            
            # 並列処理でドキュメントを処理
            import asyncio
            
            tasks = []
            for doc in documents:
                task = asyncio.create_task(
                    self.process_document(doc, chunk_size, overlap)
                )
                tasks.append(task)
            
            # 全てのタスクを実行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_chunks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Error processing document {i}: {result}")
                    continue
                
                if hasattr(result, 'is_success') and result.is_success():
                    chunks = result.unwrap()
                    all_chunks.extend(chunks)
                else:
                    print(f"Failed to process document {i}: {result}")
            
            print(f"Successfully processed {len(all_chunks)} total chunks from {len(documents)} documents")
            return Result.success(all_chunks)
            
        except Exception as e:
            print(f"Error processing documents: {e}")
            return Result.failure(e)


class HybridSearchService:
    """ハイブリッド検索サービス"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_search_service: VectorSearchService,
        document_processing_service: DocumentProcessingService,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_search_service = vector_search_service
        self._document_processing_service = document_processing_service
    
    async def search(self, query: RAGQuery) -> Result[List[SearchResult], Exception]:
        """ハイブリッド検索を実行"""
        try:
            results = []
            
            # 1. ベクター検索
            if "vector" in query.search_types:
                vector_results = await self._vector_search(query.text)
                if vector_results.is_success():
                    results.extend(vector_results.unwrap())
            
            # 2. リアルタイム検索（必要な場合）
            if "realtime" in query.search_types:
                # ベクター検索の結果が不十分な場合にリアルタイム検索を実行
                if len(results) < query.max_results:
                    realtime_results = await self._realtime_search(query.text)
                    if realtime_results.is_success():
                        results.extend(realtime_results.unwrap())
            
            # 3. 結果をスコアでソート
            results.sort(key=lambda x: x.score.value, reverse=True)
            
            # 4. 上位結果を返す
            return Result.success(results[:query.max_results])
            
        except Exception as e:
            return Result.failure(e)
    
    async def _vector_search(self, query_text: QueryText) -> Result[List[SearchResult], Exception]:
        """ベクター検索"""
        try:
            # クエリのエンベッディングを生成
            embedding_result = await self._embedding_service.embed_text(query_text.value)
            if embedding_result.is_failure():
                return Result.failure(embedding_result._value)
            
            query_embedding = embedding_result.unwrap()
            
            # ベクター検索を実行
            search_result = await self._vector_search_service.search_similar(
                query_embedding, limit=5
            )
            
            return search_result
            
        except Exception as e:
            return Result.failure(e)
    
    async def _realtime_search(self, query_text: QueryText) -> Result[List[SearchResult], Exception]:
        """リアルタイム検索（無効化）"""
        # リアルタイム検索機能は廃止されました
        return Result.success([])


class RAGOrchestratorService:
    """RAG オーケストレーターサービス"""
    
    def __init__(
        self,
        hybrid_search_service: HybridSearchService,
        llm_service: LLMService,
    ) -> None:
        self._hybrid_search_service = hybrid_search_service
        self._llm_service = llm_service
    
    async def process_query(self, query: RAGQuery) -> Result[RAGResponse, Exception]:
        """クエリを処理してRAG応答を生成"""
        try:
            start_time = datetime.now()
            
            # 1. ハイブリッド検索を実行
            search_result = await self._hybrid_search_service.search(query)
            if search_result.is_failure():
                return Result.failure(search_result._value)
            
            search_results = search_result.unwrap()
            
            # 2. LLMで回答を生成
            answer_result = await self._llm_service.generate_answer(
                query.text.value, search_results
            )
            
            if answer_result.is_failure():
                return Result.failure(answer_result._value)
            
            answer = answer_result.unwrap()
            
            # 3. 応答時間を計算
            end_time = datetime.now()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # 4. RAG応答を構築
            response = RAGResponse(
                query_id=query.id,
                answer=answer,
                sources=search_results,
                response_time_ms=response_time_ms,
            )
            
            return Result.success(response)
            
        except Exception as e:
            return Result.failure(e) 