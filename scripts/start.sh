#!/bin/bash
# Hybrid RAG System 起動スクリプト

set -e

echo "🚀 Hybrid RAG System を起動しています..."

# 環境変数の確認
echo "📋 環境変数を確認中..."
export QDRANT_HOST=${QDRANT_HOST:-"localhost"}
export QDRANT_PORT=${QDRANT_PORT:-"6333"}
export OLLAMA_HOST=${OLLAMA_HOST:-"http://localhost:11434"}
export OLLAMA_MODEL=${OLLAMA_MODEL:-"llama3.2:3b"}
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-"all-MiniLM-L6-v2"}
export WIKIPEDIA_LANG=${WIKIPEDIA_LANG:-"ja"}

# Dockerコンテナの起動
echo "🐳 Dockerコンテナを起動中..."
docker compose up -d qdrant ollama

# サービスの起動待機
echo "⏳ サービスの起動を待機中..."
sleep 10

# Ollamaモデルのプル（M3 Mac対応）
echo "🤖 Ollamaモデルの準備中..."
echo "（M3 Macではモデル初回使用時に自動ダウンロードされます）"
# docker exec sample-rag-ollama-1 ollama pull "${OLLAMA_MODEL}"

# 依存関係のインストール
echo "📦 依存関係をインストール中..."
uv sync

# APIサーバーの起動
echo "🌐 RAG APIサーバーを起動中..."
echo "API URL: http://localhost:8000"
echo "Health Check: http://localhost:8000/api/v1/health"
echo "Docs: http://localhost:8000/docs"

uv run python -m src.main
