#!/bin/bash
# RAG API テストスクリプト

set -e

BASE_URL="http://localhost:8000"

echo "🧪 RAG APIのテストを開始します..."

# ヘルスチェック
echo "🔍 ヘルスチェック..."
curl -s "${BASE_URL}/api/v1/health" | jq '.'

# 文書取り込み（労働法関連）
echo "📚 文書取り込み（労働法関連）..."
curl -s -X POST "${BASE_URL}/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{"query": "労働基準法 36協定", "limit": 3}' | jq '.'

echo "⏳ インデックス作成を待機中..."
sleep 5

# 36協定に関する質問
echo "❓ 36協定に関する質問..."
curl -s -X POST "${BASE_URL}/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "36協定の最大の労働超過可能時間は何時間？"}' | jq '.'

# 検索テスト
echo "🔍 検索テスト..."
curl -s -X POST "${BASE_URL}/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "労働時間", "search_types": ["vector", "realtime"], "max_results": 3}' | jq '.'

echo "✅ APIテストが完了しました！"
