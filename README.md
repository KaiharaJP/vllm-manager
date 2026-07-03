# vLLM Manager

vLLM サーバーを Web UI から管理するためのローカル管理アプリです。モデル登録、管理者限定の Hugging Face ダウンロード、vLLM 起動/停止、リアルタイムメトリクス、LiteLLM のユーザー/チーム/API キー/予算管理をまとめて扱います。

## アーキテクチャ

通常起動では、vLLM は `backend` コンテナ内のサブプロセスとして起動します。LiteLLM とメトリクス取得先もこの backend 管理の vLLM に向きます。

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Next.js    │────▶│  FastAPI    │────▶│  vLLM       │
│  frontend   │     │  backend    │     │  subprocess │
│  :3000      │     │  :8000      │     │  :8001      │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌───────────┐             ┌────────────┐
        │ LiteLLM   │────────────▶│ PostgreSQL │
        │ :4000     │             │ LiteLLM DB │
        └───────────┘             └────────────┘
```

`docker-compose.yml` には `standalone-vllm` profile の `vllm` サービスも残していますが、通常は起動しません。手動検証などで vLLM コンテナを直接立てたい場合だけ使います。

## 主な機能

- **ログイン/RBAC**: `admin` と `user` ロール。サーバー操作、モデルダウンロード、LiteLLM 管理は admin のみ。
- **サーバー管理**: 登録済みモデルを選び、vLLM を起動/停止/再起動。
- **起動パラメータ**: コンテキスト長、最大同時リクエスト数、GPU メモリ利用率、テンソル並列数、起動時ダウンロード有無を UI から指定。
- **モデル管理**: admin が Hugging Face `repo_id` を登録し、非同期ジョブでダウンロード。
- **リアルタイムイベント**: `/ws/metrics` と `/ws/events` でメトリクス、モデルダウンロード、サーバージョブ、LiteLLM 変更イベントを配信。
- **LiteLLM 管理**: vLLM Manager 画面から LiteLLM のユーザー、チーム、virtual key、予算、RPM/TPM、利用ログを扱う。
- **永続化**: vLLM Manager の設定/ユーザー/ジョブは `vllm-data` volume、HF キャッシュは `hf-cache` volume、LiteLLM 管理データは Postgres の `litellm-db` volume に保存。

## 前提条件

- Docker & Docker Compose
- NVIDIA GPU
- NVIDIA Container Toolkit
- Hugging Face gated model を使う場合は Hugging Face access token

## クイックスタート

```bash
cp .env.example .env
# .env を編集する
docker compose up -d --build
```

アクセス先のデフォルトは次の通りです。

- vLLM Manager UI: `http://hinton.kanazawa-it.ac.jp:13000`
- FastAPI backend: `http://hinton.kanazawa-it.ac.jp:18000`
- backend 管理 vLLM OpenAI API: `http://hinton.kanazawa-it.ac.jp:18001`
- LiteLLM Proxy: `http://hinton.kanazawa-it.ac.jp:14000`

### どのポートを使うべきか（重要）

`18001` に統一されたわけではありません。用途ごとに入口が異なります。

| 用途 | 推奨接続先 | 補足 |
|------|------------|------|
| OpenAI 形式を backend 経由で使う | `http://hinton.kanazawa-it.ac.jp:18000/v1` | 管理アプリのプロキシを通る入口 |
| OpenAI 形式を LiteLLM 経由で使う | `http://hinton.kanazawa-it.ac.jp:14000/v1` | チーム運用・キー管理向け |
| Claude Code（Anthropic 形式）で使う | `http://hinton.kanazawa-it.ac.jp:14000` | `ANTHROPIC_BASE_URL` に設定 |
| vLLM を直接叩く | `http://hinton.kanazawa-it.ac.jp:18001/v1` | 直接接続。管理プロキシは経由しない |

普段の運用は **`18000`（backend）または `14000`（LiteLLM）** を推奨します。  
`18001` は「vLLM 直接接続」をしたいときだけ使ってください。

初期ログインは `.env` の `VLLM_MANAGER_ADMIN_USER` / `VLLM_MANAGER_ADMIN_PASSWORD` です。デフォルトは `admin` / `admin` なので、実運用では必ず変更してください。

## .env に入れる情報

`.env.example` を `.env` にコピーして使います。`.env` には API キーやパスワードが入るため、Git にコミットしないでください。

### vLLM Manager 認証

| 変数 | 必須 | 説明 | 例/デフォルト |
|------|------|------|---------------|
| `VLLM_MANAGER_ADMIN_USER` | 推奨 | 初回起動時に作られる vLLM Manager 管理者ユーザー名。`/app/data/users.json` が既にある場合は既存ユーザーが優先されます。 | `admin` |
| `VLLM_MANAGER_ADMIN_PASSWORD` | 推奨 | 初回管理者のパスワード。bcrypt hash 化されて `vllm-data` volume に保存されます。 | `admin` |

`VLLM_MANAGER_ADMIN_PASSWORD=admin` のまま公開ネットワークに出さないでください。既に初期ユーザーが作られた後に変更したい場合は、UI で別 admin を作るか、`vllm-data` volume 内の `users.json` を再作成します。

### アプリケーション公開ポート

| 変数 | 必須 | 説明 | 例/デフォルト |
|------|------|------|---------------|
| `FRONTEND_PORT` | 任意 | ホスト側の Next.js UI ポート。 | `13000` |
| `BACKEND_PORT` | 任意 | ホスト側の FastAPI ポート。 | `18000` |
| `VLLM_HOST_PORT` | 任意 | ホスト側に公開する vLLM OpenAI API ポート。 | `18001` |
| `VLLM_PORT` | 任意 | コンテナ内の vLLM ポート。通常は変えません。 | `8001` |
| `LITELLM_PORT` | 任意 | ホスト側の LiteLLM Proxy ポート。 | `14000` |

`docker-compose.yml` 側のデフォルトは `FRONTEND_PORT=13000`, `BACKEND_PORT=18000`, `VLLM_HOST_PORT=18001`, `LITELLM_PORT=14000` です。古い README の `3000/8000/4000` ではなく、ホストからはこのポートを見ます。

### Hugging Face

| 変数 | 必須 | 説明 | 例/デフォルト |
|------|------|------|---------------|
| `HF_TOKEN` | gated model では必須 | Hugging Face の access token。backend と standalone vLLM の両方に `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` として渡されます。 | `hf_xxx` |
| `DOWNLOAD_WORKERS` | 任意 | モデルダウンロード時の並列数。大きすぎるとネットワークやディスクに負荷がかかります。 | `8` |

`HF_TOKEN` は UI に平文で返しません。gated model を使う場合は Hugging Face 側でライセンス同意を済ませた token を設定してください。

### LiteLLM

| 変数 | 必須 | 説明 | 例/デフォルト |
|------|------|------|---------------|
| `LITELLM_MASTER_KEY` | 必須 | vLLM Manager backend が LiteLLM Admin API を呼ぶための master key。推論 API の Bearer token としても使えます。 | `sk-vllm-default-key` |
| `LITELLM_UI_USERNAME` | 任意 | LiteLLM 標準 Admin UI のログインユーザー。vLLM Manager 統合 UI とは別です。 | `admin` |
| `LITELLM_UI_PASSWORD` | 任意 | LiteLLM 標準 Admin UI のログインパスワード。 | `admin` |
| `LITELLM_DB_USER` | 任意 | LiteLLM 用 Postgres ユーザー。 | `litellm` |
| `LITELLM_DB_PASSWORD` | 任意 | LiteLLM 用 Postgres パスワード。 | `litellm` |
| `LITELLM_DB_NAME` | 任意 | LiteLLM 用 Postgres DB 名。 | `litellm` |

LiteLLM の virtual keys、users、teams、budgets、spend logs は `litellm-db` volume の Postgres に保存されます。`LITELLM_MASTER_KEY` は `sk-` で始まる値にしてください。

### vLLM 起動デフォルト

| 変数 | 必須 | 説明 | 例/デフォルト |
|------|------|------|---------------|
| `VLLM_MAX_MODEL_LEN` | 任意 | vLLM 起動フォームの初期値として使いたい最大コンテキスト長。現状の保存設定がある場合は保存値が優先されます。 | `8192` |
| `VLLM_MAX_NUM_SEQS` | 任意 | 最大同時リクエスト数の初期値。 | `256` |
| `VLLM_GPU_MEMORY_UTILIZATION` | 任意 | GPU メモリ利用率の初期値。 | `0.9` |
| `VLLM_TENSOR_PARALLEL_SIZE` | 任意 | テンソル並列数の初期値。 | `1` |

現在の vLLM 起動設定は `vllm-data` volume の `config.json` に保存されます。UI で変更した値が次回以降も使われます。

### Docker / 社内 HTTP プロキシ（任意）

| 変数 | 必須 | 説明 | 例/デフォルト |
|------|------|------|---------------|
| `DOCKER_COMPOSE_PROJECT_NAME` | 任意 | Docker Compose のプロジェクト名。 | `vllm-manager` |
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` | 通常不要 | 社内 proxy 経由で Hugging Face 等を取得する場合のみ `.env` に設定。未設定（空）で問題ありません。 | 空 |

### Blackwell + NVFP4 モデル（例: `*-NVFP4`）

| 要件 | 説明 |
|------|------|
| CUDA Toolkit | **12.9 以上**（backend イメージに同梱。FlashInfer が SM120 向け FP4 カーネルを JIT コンパイルするため） |
| GPU | Blackwell (compute capability 12.x) |
| ホスト CUDA マウント | 通常不要。上書きする場合は 12.9+ の Toolkit のみ（`docker-compose.override.yml`） |

MTP（`speculative_config`）は別要件。起動失敗時は `/app/data/vllm.log` の `flashinfer` / `nvcc` 行を確認してください。

## データの保存先

- `vllm-data`: vLLM Manager の `users.json`, `config.json`, `download_jobs.json`, `models.json`, `vllm.pid`, `vllm.log`
- `hf-cache`: Hugging Face model cache
- `litellm-db`: LiteLLM の Postgres データ
- ブラウザ `localStorage`: ログインセッション token のみ

### モデルデータの実体パス

モデル本体（Hugging Face キャッシュ）は `hf-cache` volume に保存されます。

- backend コンテナ内: `/root/.cache/huggingface`
- ホスト側実体: `/var/lib/docker/volumes/vllm-manager_hf-cache/_data`

主な配置例:

- `<HF_HOME>/models--<org>--<repo>/`（新しいレイアウト）
- `<HF_HOME>/hub/models--<org>--<repo>/`（互換レイアウト）

`HF_HOME` は backend で `/root/.cache/huggingface` に固定しています（`docker-compose.yml`）。

backend を再起動するとブラウザの token は残りますが、現在の簡易セッションは backend メモリ上にあるため、再ログインが必要になることがあります。

## 使い方

### 1. ログイン

UI にアクセスし、`.env` の `VLLM_MANAGER_ADMIN_USER` / `VLLM_MANAGER_ADMIN_PASSWORD` でログインします。admin だけがサーバー操作、モデル管理、LiteLLM 管理を実行できます。

### 2. モデル登録とダウンロード

`モデル管理` タブで Hugging Face `repo_id` を登録します。登録後、`ダウンロード` を押すと backend で非同期ジョブが開始され、WebSocket 経由で進捗が画面に反映されます。

### 3. vLLM 起動

`サーバー管理` タブでモデル、コンテキスト長、最大同時リクエスト数、GPU メモリ利用率、テンソル並列数を指定して起動します。未ダウンロードモデルを起動する場合は「起動前にモデルキャッシュを確認/ダウンロードする」を有効にできます。

### 4. LiteLLM 管理

`ユーザー/APIキー` タブで vLLM Manager ユーザーを作成し、LiteLLM の virtual key を発行します。key 発行時に `models`, `max_budget`, `budget_duration`, `rpm_limit`, `tpm_limit` を指定できます。

### 5. 推論

LiteLLM 経由で呼び出す例です。

```bash
curl http://hinton.kanazawa-it.ac.jp:14000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vllm-default-key" \
  -d '{
    "model": "vllm-local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

backend 管理 vLLM を直接叩く場合は `http://hinton.kanazawa-it.ac.jp:18001/v1/chat/completions` を使います。

### Claude Code から LiteLLM 経由で使う（Anthropic 形式）

Claude Code は `ANTHROPIC_BASE_URL` で LLM gateway を指定できます。  
この構成では Claude Code が Anthropic Messages 形式で LiteLLM を呼び、LiteLLM が vLLM（OpenAI 互換）へ変換します。

```bash
export ANTHROPIC_BASE_URL="http://hinton.kanazawa-it.ac.jp:14000"
export ANTHROPIC_AUTH_TOKEN="sk-vllm-default-key"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-vllm-local"

claude
```

接続確認は次でできます。

```bash
curl http://hinton.kanazawa-it.ac.jp:14000/v1/messages \
  -H "Authorization: Bearer sk-vllm-default-key" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-vllm-local",
    "max_tokens": 64,
    "messages": [{"role":"user","content":"hello"}]
  }'
```

## API エンドポイント

### 認証

- `POST /api/auth/login` - vLLM Manager にログイン
- `GET /api/auth/me` - 現在のログインユーザー
- `GET /api/users` - Manager ユーザー一覧 admin only
- `POST /api/users` - Manager ユーザー作成/更新 admin only

### サーバー管理

- `GET /api/status` - vLLM サーバー状態
- `POST /api/start` - vLLM 起動 admin only
- `POST /api/stop` - vLLM 停止 admin only
- `POST /api/restart` - vLLM 再起動 admin only
- `GET /api/log` - vLLM ログ admin only

### モデル管理

- `GET /api/models` - 登録済みモデル一覧
- `POST /api/models` - モデル登録 admin only
- `GET /api/model-downloads` - ダウンロードジョブ一覧 admin only
- `POST /api/model-downloads` - ダウンロードジョブ開始 admin only
- `GET /api/context-presets` - コンテキスト長プリセット

### LiteLLM

- `GET /api/litellm/status` - LiteLLM health
- `GET /api/litellm/keys` - virtual key 一覧 admin only
- `POST /api/litellm/keys` - virtual key 発行 admin only
- `POST /api/litellm/keys/delete` - virtual key 削除 admin only
- `GET /api/litellm/users` - LiteLLM user 一覧 admin only
- `POST /api/litellm/users` - LiteLLM user 作成 admin only
- `GET /api/litellm/teams` - LiteLLM team 一覧 admin only
- `POST /api/litellm/teams` - LiteLLM team 作成 admin only
- `GET /api/litellm/spend` - spend logs admin only

### WebSocket

- `WS /ws/metrics` - 互換パス。メトリクスとイベントを受信
- `WS /ws/events` - イベント購読用パス

イベントは `type`, `timestamp`, `data`, `message`, `actor` を持つ envelope 形式です。主な `type` は `metrics`, `model_download`, `server_job`, `model_registered`, `user_updated`, `litellm_key_updated`, `error` です。

## standalone vLLM profile

通常は使いません。vLLM コンテナを直接起動したい場合だけ、次のように profile を指定します。

```bash
docker compose --profile standalone-vllm up -d vllm
```

この profile を使う場合、backend 管理の vLLM とポートが競合しないようにしてください。

## トラブルシュート

- **ログインできない**: 初回ユーザーは `vllm-data` volume の `users.json` に作られます。`.env` を変えても既存 `users.json` がある場合は上書きされません。
- **gated model が落ちる**: `HF_TOKEN` が設定されているか、Hugging Face 側で対象モデルの利用許諾が済んでいるか確認してください。
- **LiteLLM のキー管理が動かない**: `litellm-db` が起動しているか、`LITELLM_MASTER_KEY` が backend と LiteLLM で一致しているか確認してください。
- **メトリクスが出ない**: vLLM が起動して `/metrics` を返すまで待ってください。起動直後やモデルロード中は空になることがあります。
- **ポートが違う**: ホストから見るポートは `.env` の `FRONTEND_PORT`, `BACKEND_PORT`, `VLLM_HOST_PORT`, `LITELLM_PORT` です。

## プロジェクト構造

```text
vllm-manager/
├── app/
│   ├── auth.py              # Manager ログイン/RBAC
│   ├── event_bus.py         # WebSocket event bus
│   ├── litellm_client.py    # LiteLLM Admin API wrapper
│   ├── main.py              # FastAPI routes
│   ├── metrics_scraper.py   # vLLM metrics scraper
│   ├── model_manager.py     # model catalog / download jobs
│   └── server_manager.py    # backend-managed vLLM process
├── frontend/
│   └── src/
│       ├── app/
│       ├── components/
│       ├── hooks/
│       ├── lib/
│       └── types/
├── config/
│   └── litellm_config.yaml
├── docker-compose.yml
├── Dockerfile.backend
├── IMPLEMENTATION_PHASES.md
└── requirements.txt
```

## 開発

```bash
# バックエンド
pip install -r requirements.txt
VLLM_MANAGER_DATA_DIR=/tmp/vllm-manager-data uvicorn app.main:app --reload --port 8000

# フロントエンド
cd frontend
npm install
npm run dev
```

ローカル開発では Docker Compose とポートが違う場合があります。`NEXT_PUBLIC_API_URL` と `NEXT_PUBLIC_LITELLM_URL` を環境に合わせてください。

## 注意

このアプリは GPU サーバーの管理操作、モデルダウンロード、API キー発行を行います。外部公開する場合は、強い管理者パスワード、ネットワーク制限、TLS、バックアップ、監査ログ運用を必ず検討してください。

## ライセンス

MIT
