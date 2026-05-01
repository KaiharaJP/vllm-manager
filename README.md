# vLLM Manager

vLLM サーバーを Web UI から管理するアプリケーション。モデル選択、コンテキスト長設定、リアルタイムモニタリング、LiteLLM 認証対応。

## アーキテクチャ

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Next.js    │────▶│  FastAPI    │────▶│  vLLM       │
│  (Frontend) │     │  (Backend)  │     │  (Inference)│
│  :3000      │     │  :8000      │     │  :8001      │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  LiteLLM    │
                    │  (Proxy)    │
                    │  :4000      │
                    └─────────────┘
```

## 機能

- **サーバー管理**: モデル選択、起動/停止/再起動
- **コンテキスト長設定**: 4K/8K/32K/64K/128K のプリセット選択
- **スロット数設定**: 最大同時リクエスト数のスライダー調整
- **リアルタイムモニタリング**: GPU メモリ使用量、リクエスト数、スループット、トークン処理速度
- **LiteLLM 認証**: API キー認証の ON/OFF トグル
- **ログ表示**: vLLM サーバーログのリアルタイム表示

## 前提条件

- Docker & Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit

## クイックスタート

```bash
# 1. リポジトリをクローン
git clone <repository>
cd vllm-manager

# 2. 環境変数設定
cp .env.example .env
# .env を編集 (必要に応じて)

# 3. 起動
docker compose up -d --build

# 4. Web UI にアクセス
# http://localhost:3000
```

## 環境変数

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `VLLM_PORT` | vLLM サーバーポート | 8001 |
| `BACKEND_PORT` | FastAPI ポート | 8000 |
| `FRONTEND_PORT` | Next.js ポート | 3000 |
| `LITELLM_MASTER_KEY` | LiteLLM API キー | sk-vllm-default-key |
| `HF_TOKEN` | HuggingFace トークン | (なし) |

## API エンドポイント

### サーバー管理
- `GET /api/status` - サーバー状態
- `POST /api/start` - サーバー起動
- `POST /api/stop` - サーバー停止
- `POST /api/restart` - サーバー再起動

### 設定
- `GET /api/config` - 現在設定
- `GET /api/models` - 利用可能モデル
- `GET /api/context-presets` - コンテキスト長プリセット
- `GET /api/log` - サーバーログ

### WebSocket
- `WS /ws/metrics` - リアルタイムメトリクス

## 利用可能なモデル

- Llama 3.3 70B Instruct
- Llama 3.1 8B Instruct
- Qwen 2.5 72B Instruct
- Qwen 2.5 7B Instruct
- Mistral 7B Instruct v0.3
- Phi-3 Mini 4K Instruct
- Gemma 2 27B IT
- Gemma 2 9B IT

## 使用例

### curl で API を叩く

```bash
# サーバー状態確認
curl http://localhost:8000/api/status

# サーバー起動
curl -X POST http://localhost:8000/api/start \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-llama/Llama-3.1-8B-Instruct", "context_length": 8192}'

# LiteLLM 経由で推論
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vllm-default-key" \
  -d '{
    "model": "vllm-local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## プロジェクト構造

```
vllm-manager/
├── app/                          # FastAPI バックエンド
│   ├── main.py                   # API エンドポイント
│   ├── server_manager.py         # vLLM サーバー管理
│   └── metrics_scraper.py        # Prometheus metrics スクラッパー
├── frontend/                     # Next.js フロントエンド
│   ├── src/
│   │   ├── app/                  # ページ
│   │   ├── components/           # コンポーネント
│   │   ├── hooks/                # カスタムフック
│   │   ├── lib/                  # API クライアント
│   │   └── types/                # TypeScript 型定義
│   └── Dockerfile
├── config/                       # 設定ファイル
│   └── litellm_config.yaml
├── docker-compose.yml
├── Dockerfile.backend
└── requirements.txt
```

## 開発

```bash
# バックエンド開発
cd vllm-manager
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# フロントエンド開発
cd frontend
npm install
npm run dev
```

## ライセンス

MIT
