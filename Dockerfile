# Python 3.13 slim image
FROM python:3.13-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージをインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uvをインストール
RUN pip install uv

# プロジェクトファイルをコピー
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# 依存関係をインストール
RUN uv sync --frozen

# ポート8000を公開
EXPOSE 8000

# ログディレクトリを作成
RUN mkdir -p /app/logs

# アプリケーションを起動
CMD ["uv", "run", "python", "-m", "src.main"] 