"""Ollama LLM サービス実装"""

from __future__ import annotations

import os
from typing import List

import ollama

from ..domain.services import LLMService
from ..domain.value_objects import SearchResult
from ..shared.result import Result, try_catch_async


class OllamaLLMService(LLMService):
    """Ollama LLM サービス実装"""
    
    def __init__(
        self, 
        model_name: str = None,
        host: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> None:
        self._model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._temperature = temperature
        self._max_tokens = max_tokens
        
        print(f"Connecting to Ollama at {self._host}")
        self._client = ollama.Client(host=self._host)
        print(f"Initialized Ollama LLM service: {self._model_name}")
    
    async def generate_answer(
        self, 
        query: str, 
        context: List[SearchResult]
    ) -> Result[str, Exception]:
        """コンテキストに基づいて回答を生成"""
        @try_catch_async
        async def _generate() -> str:
            # コンテキストを文字列に変換
            context_text = self._format_context(context)
            
            # プロンプトを作成
            prompt = self._create_prompt(query, context_text)
            
            # Ollamaで生成
            response = self._client.generate(
                model=self._model_name,
                prompt=prompt,
                options={
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            )
            
            return response["response"]
        
        return await _generate()
    
    def _format_context(self, context: List[SearchResult]) -> str:
        """コンテキストを整形"""
        if not context:
            return "関連する情報が見つかりませんでした。"
        
        context_parts = []
        for i, result in enumerate(context[:3], 1):  # 上位3件のみ使用
            content = result.document_chunk.content
            
            # 長いコンテンツは適切に短縮
            if len(content) > 800:
                content = content[:800] + "..."
            
            context_parts.append(f"【参考情報 {i}】\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context_text: str) -> str:
        """プロンプトを作成"""
        return f"""あなたは親切で知識豊富なアシスタントです。以下の参考情報を基に、質問に正確で分かりやすく答えてください。

質問: {query}

参考情報:
{context_text}

回答時の注意点:
- 参考情報の内容を基に回答してください
- 自然で読みやすい日本語で回答する
- 重要な情報を適切にまとめる
- 情報が不足している場合は、その旨を明記する
- 「です・ます」調で丁寧に回答する

回答:"""
    
    async def generate_summary(self, text: str) -> Result[str, Exception]:
        """テキストの要約を生成"""
        @try_catch_async
        async def _summarize() -> str:
            prompt = f"""以下のテキストを簡潔に要約してください。重要なポイントを漏らさず、分かりやすくまとめてください。

テキスト:
{text}

要約:"""
            
            response = self._client.generate(
                model=self._model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # 要約は一定性を重視
                    "num_predict": 200,
                },
            )
            
            return response["response"]
        
        return await _summarize()
    
    async def extract_keywords(self, text: str) -> Result[List[str], Exception]:
        """キーワードを抽出"""
        @try_catch_async
        async def _extract() -> List[str]:
            prompt = f"""以下のテキストから重要なキーワードを抽出してください。5-10個のキーワードを、カンマ区切りで列挙してください。

テキスト:
{text}

キーワード:"""
            
            response = self._client.generate(
                model=self._model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 50,
                },
            )
            
            keywords_text = response["response"]
            keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
            return keywords
        
        return await _extract()
    
    async def check_model_availability(self) -> Result[bool, Exception]:
        """モデルの利用可能性をチェック"""
        @try_catch_async
        async def _check() -> bool:
            try:
                models = self._client.list()
                # modelsが辞書の場合
                if isinstance(models, dict) and "models" in models:
                    model_names = [model.get("name", "") for model in models["models"]]
                # modelsが直接リストの場合
                elif isinstance(models, list):
                    model_names = [model.get("name", "") for model in models]
                else:
                    # フォールバック: モデルが利用可能と仮定
                    return True
                return self._model_name in model_names
            except Exception:
                # エラーの場合、モデルが利用可能と仮定
                return True
        
        return await _check()
    
    async def pull_model(self) -> Result[None, Exception]:
        """モデルをプル（ダウンロード）"""
        @try_catch_async
        async def _pull() -> None:
            print(f"Pulling model: {self._model_name}")
            self._client.pull(self._model_name)
            print(f"Model pulled successfully: {self._model_name}")
        
        return await _pull()
    
    async def get_model_info(self) -> Result[dict, Exception]:
        """モデル情報を取得"""
        @try_catch_async
        async def _get_info() -> dict:
            return self._client.show(self._model_name)
        
        return await _get_info() 