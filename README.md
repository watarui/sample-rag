# Web RAG System

## 概要

Web スクレイピングを活用した RAG システムです。指定された Web サイトから情報を取得し、ベクターデータベース（Qdrant）に保存して、包括的な質問応答を実現します。

## 特徴

- **Web スクレイピング**: 指定された Web サイトから自動的にデータを取得
- **ベクター検索**: 高精度な意味検索による関連情報の抽出
- **ローカル LLM**: セキュリティを重視したローカル LLM（Ollama）を使用
- **関数型設計**: 関数型プログラミングの原則を採用
- **Clean Architecture**: ドメイン駆動設計（DDD）と Clean Architecture を採用
- **型安全性**: 厳密な型定義による開発の安全性

## 動作原理

1. **Web スクレイピング**: 指定された Web サイトから情報を取得
2. **チャンク化**: 取得したデータを検索可能なチャンクに分割
3. **ベクター化**: エンベッディングを生成してベクターデータベースに保存
4. **質問応答**: ベクター検索により関連情報を抽出し、LLM で回答を生成

## 技術スタック

- **Python 3.13**: モダンな Python 機能を活用
- **FastAPI**: 高性能な WebAPI
- **Qdrant**: ベクターデータベース
- **Ollama**: ローカル LLM
- **Docker**: コンテナ化
- **Pydantic**: データ検証と型安全性

## 使用方法

### 🚀 簡単起動（推奨）

```bash
# 一括起動スクリプト
./scripts/start.sh
```

### 📱 M3 Mac での設定

```bash
# M3 Mac用の軽量モデルを使用する場合
export OLLAMA_MODEL=llama3.2:1b

# または高性能モデル（メモリ使用量が多い）
export OLLAMA_MODEL=llama3.2:3b
```

### 🔧 手動起動

```bash
# 依存関係のインストール
uv sync

# Dockerコンテナの起動
docker compose up -d

# APIサーバーの起動
uv run python -m src.main

# 質問例
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "サイトの情報について教えて"}'
```

### 🧪 テスト実行

```bash
# APIテストの実行
./scripts/test_api.sh
```

## アーキテクチャ

```
src/
├── domain/           # ドメインモデル
├── application/      # アプリケーションサービス
├── infrastructure/   # インフラストラクチャ
├── presentation/     # プレゼンテーション層
└── shared/          # 共有コンポーネント
```

## 注意事項

- 初回起動時に Ollama モデルのダウンロードが必要です
- Web スクレイピングは robots.txt と利用規約に従って適切に実行されます
- スクレイピング対象の Web サイトは `PORTAL_URLS` 環境変数で設定してください

### 🍎 M3 Mac 特有の注意事項

- Apple Silicon GPU が自動的に使用されるため、NVIDIA GPU 設定は不要です
- 初回モデル使用時に自動的にダウンロードされます（数 GB 必要）
- メモリ使用量を抑えたい場合は `llama3.2:1b` モデルを推奨します
- Docker Compose は新しい `docker compose` コマンドを使用してください
