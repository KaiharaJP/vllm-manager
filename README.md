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
- **サーバー管理**: 登録済みモデルを選び、vLLM を起動/停止/再起動。**VRAM が許す限り chat と embedding を別インスタンスで同時起動**できます。
- **起動パラメータ**: コンテキスト長、最大同時リクエスト数、GPU メモリ利用率、テンソル並列数、起動時ダウンロード有無を UI から指定。
- **モデル管理**: admin が Hugging Face `repo_id` を登録し、用途（`chat` / `embedding`）を指定して非同期ジョブでダウンロード。
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

- backend コンテナ内: `/app/hf-cache`
- ホスト側実体: `/var/lib/docker/volumes/vllm-manager_hf-cache/_data`

主な配置例:

- `<HF_HOME>/models--<org>--<repo>/`（新しいレイアウト）
- `<HF_HOME>/hub/models--<org>--<repo>/`（互換レイアウト）

`HF_HOME` は backend で `/app/hf-cache` に固定しています（`docker-compose.yml`）。同一 Docker ボリューム `hf-cache` をマウントしており、旧パス `/root/.cache/huggingface` から移行してもデータは保持されます。

backend を再起動するとブラウザの token は残りますが、**セッショントークン**は backend メモリ上にあるため失効します。**永続 API トークン（PAT）** を使えば再起動後も HTTP API をそのまま利用できます（[セキュリティ・運用ドキュメント](docs/security-and-operations.md#12-永続-api-トークンpatと-cli) 参照）。

## 使い方

### 1. ログイン

UI にアクセスし、`.env` の `VLLM_MANAGER_ADMIN_USER` / `VLLM_MANAGER_ADMIN_PASSWORD` でログインします。admin だけがサーバー操作、モデル管理、LiteLLM 管理を実行できます。

**チャット**: ログイン後、「チャット」タブから起動中のモデルとブラウザ上で会話できます（API キーの手動設定は不要）。詳細は [docs/security-and-operations.md](docs/security-and-operations.md#16-管理画面内チャット-ui) を参照してください。

### 2. モデル登録とダウンロード

`モデル管理` タブで Hugging Face `repo_id` を登録します。登録後、`ダウンロード` を押すと backend で非同期ジョブが開始され、WebSocket 経由で進捗が画面に反映されます。

### 3. vLLM 起動

`サーバー管理` タブでモデル、コンテキスト長、最大同時リクエスト数、GPU メモリ利用率、テンソル並列数を指定して起動します。未ダウンロードモデルを起動する場合は「起動前にモデルキャッシュを確認/ダウンロードする」を有効にできます。

**複数モデル同時起動**: 「起動」するたびに新しい vLLM インスタンスが追加されます（既存は停止しません）。インスタンス名を付けると一覧で識別しやすくなります。停止は「起動中サーバー一覧」から個別に行います。

**埋め込みモデル**: モデル管理で `task_type: embedding` を選んで登録・ダウンロード後、サーバー管理から起動します。vLLM は `--runner pooling` で立ち上がり、`/v1/embeddings` で利用できます。LiteLLM 経由（`:14000/v1`）でも backend プロキシ経由で同じ API キー・モデル許可の仕組みが使えます。

任意の Hugging Face embedding モデル（例: `jinaai/jina-embeddings-v3`）を登録できます。登録時に推奨コンテキスト長・出力次元・ライセンス注意を付けておくと、起動 UI で安全に運用しやすくなります。

```bash
# 管理者 PAT（vlmk_...）で DL
export VLLM_MANAGER_URL=http://hinton.kanazawa-it.ac.jp:18000
./scripts/vllm-cli.sh models download org/your-embedding-model

# embedding インスタンス起動（context はモデル上限に合わせる）
./scripts/vllm-cli.sh start org/your-embedding-model \
  --context-length 8192 --task-type embedding --no-download

# 疎通確認（1792 次元などは smoke test の response_preview で確認）
./scripts/vllm-cli.sh smoke-test <instance_id>
```

LiteLLM 経由のクエリ埋め込み例:

```bash
curl http://hinton.kanazawa-it.ac.jp:14000/v1/embeddings \
  -H "Authorization: Bearer sk-vllm-default-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"org/your-embedding-model","input":"検索クエリ"}'
```

大規模 chat モデルと同じ GPU に常駐できない場合は、検索時だけ chat を停止して embed を起動する運用に切り替えてください（GPU 番号・同時常駐可否は環境依存です）。

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

### 6. CLI / HTTP API でサーバー操作（自動化）

ブラウザを使わず curl やシェルスクリプトからサーバー起動/停止・モデルダウンロードができます。

**初回のみ** — 永続 API トークン（PAT）を発行してローカルに保存:

```bash
./scripts/vllm-cli.sh token create --name my-automation \
  --username admin --password 'your-password'
# → ~/.config/vllm-manager/token に保存（chmod 600）
```

**以降** — ログイン不要:

```bash
export VLLM_MANAGER_URL=http://hinton.kanazawa-it.ac.jp:18000  # 必要に応じて

./scripts/vllm-cli.sh status
./scripts/vllm-cli.sh models download Qwen/Qwen2.5-7B-Instruct
./scripts/vllm-cli.sh start Qwen/Qwen2.5-7B-Instruct --context-length 32768
./scripts/vllm-cli.sh smoke-test default
./scripts/vllm-cli.sh stop
```

curl で直接叩く例:

```bash
TOKEN="vlmk_..."  # PAT（scripts/vllm-cli.sh token create で取得）

curl -X POST "$VLLM_MANAGER_URL/api/start" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"Qwen/Qwen2.5-7B-Instruct"}'
```

PAT は LiteLLM 推論用キー（`sk-...`）とは別物です。詳細は [docs/security-and-operations.md](docs/security-and-operations.md) の「12. 永続 API トークン（PAT）と CLI」を参照してください。

### 7. バックアップ/リストア

```bash
# ユーザー・APIキー・監査ログ・LiteLLM DB をバックアップ（./backups/ に保存）
./scripts/backup-vllm-data.sh

# 復元（先に対象サービスを停止）
docker compose stop backend
./scripts/restore-vllm-data.sh vllm-data ./backups/vllm-data-<timestamp>.tar.gz
docker compose up -d backend
```

詳細は [docs/security-and-operations.md](docs/security-and-operations.md) の「15. バックアップ/リストア」を参照してください。

## API エンドポイント

### 認証

- `POST /api/auth/login` - vLLM Manager にログイン
- `GET /api/auth/me` - 現在のログインユーザー
- `GET /api/auth/me/tokens` - 永続 API トークン（PAT）一覧
- `POST /api/auth/me/tokens` - PAT 発行（生キーは一度だけ返る）
- `DELETE /api/auth/me/tokens/{token_id}` - PAT 失効
- `GET /api/chat/models` - チャット UI 用の利用可能モデル一覧
- `POST /api/chat/completions` - チャット UI 用の補完（SSE ストリーミング対応）
- `GET /api/users` - Manager ユーザー一覧 admin only
- `POST /api/users` - Manager ユーザー作成/更新 admin only

### サーバー管理

- `GET /api/status` - vLLM サーバー状態（後方互換: default または最初の稼働インスタンス）
- `GET /api/instances` - 管理対象 vLLM インスタンス一覧 admin only
- `POST /api/start` - vLLM 起動 admin only（`create_new_instance`, `task_type`, `instance_name` 対応）
- `POST /api/stop` - default インスタンス停止 admin only
- `POST /api/instances/stop` - 指定 instance_id を停止 admin only
- `POST /api/restart` - vLLM 再起動 admin only
- `POST /api/instances/{instance_id}/smoke-test` - 最小リクエストで実際の応答を検証 admin only
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
├── skills/
│   └── vllm-manager/        # Cursor Agent Skill（他プロジェクトへコピー可）
├── docker-compose.yml
├── Dockerfile.backend
├── IMPLEMENTATION_PHASES.md
└── requirements.txt
```

## 開発

### Cursor Skills パッケージ（他プロジェクト向け）

vLLM Manager の操作手順・API・CLI を Cursor Agent が参照できる Skill として同梱しています。

```bash
# 個人用（全プロジェクトで有効）
cp -r skills/vllm-manager ~/.cursor/skills/

# または特定リポジトリのみ
cp -r skills/vllm-manager /path/to/other-project/.cursor/skills/
```

- `skills/vllm-manager/SKILL.md` — エージェント向けクイックリファレンス
- `skills/vllm-manager/scripts/vllm-cli.sh` — PAT 認証付き CLI（`curl` + `jq`）
- 詳細: [skills/README.md](skills/README.md)

推論用の **`sk-` キーは従来どおり** `:14000` で利用。Skill は主にサーバー起動/停止・モデルDLの自動化と、エージェントへの接続情報の共有用です。

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

## セキュリティ・運用（vLLM Manager 本体）

| 機能 | 説明 | 関連 env |
|------|------|----------|
| WebSocket 認証 | `/ws/metrics` はログイン token 必須（`?token=`） | — |
| セッション TTL | 管理 UI トークンの有効期限 | `VLLM_MANAGER_SESSION_TTL_SEC`（既定 86400 秒） |
| ログイン試行制限 | 連続失敗で一時ロック | `VLLM_MANAGER_LOGIN_*` |
| 監査ログ | 管理操作を `vllm-data/audit.log` に JSONL 永続化 | — |
| vLLM 自動復旧 | backend 再起動時、`auto_restore=true` のインスタンスを再起動 | 手動停止で `auto_restore=false` |
| CORS | 許可オリジンを env で制限可能 | `VLLM_MANAGER_CORS_ORIGINS` |
| 弱いデフォルト認証情報の警告 | admin パスワードや `LITELLM_MASTER_KEY` が既定値のままだと管理者にバナー表示 | — |
| 弱いパスワードの強制変更 | 推測されやすいパスワードのアカウントはログイン直後に変更必須（全ロール対象） | — |
| ヘルス失敗通知 | インスタンスが3回連続で unhealthy になるとイベント発行（自動再起動はしない） | — |
| non-root 実行 | backend コンテナは既定 UID/GID 1000 の `vllmapp` で動作 | `APP_UID` / `APP_GID`（build args） |
| GPU VRAM 事前チェック | 起動前に空き VRAM を確認し不足時は起動を中止（既存実装） | — |
| 監査ログローテーション | 上限サイズ超過で自動ローテーション、古い世代は削除 | `VLLM_MANAGER_AUDIT_LOG_MAX_BYTES` / `VLLM_MANAGER_AUDIT_LOG_BACKUPS` |

管理者向け: `GET /api/audit-log?limit=100` で監査ログを取得できます。

詳細・各変更の影響は [docs/security-and-operations.md](docs/security-and-operations.md) を参照してください。特に **non-root 実行化はコンテナ運用に関わる変更**のため、適用前に必ずドキュメントの注意点を確認してください。

## 注意

このアプリは GPU サーバーの管理操作、モデルダウンロード、API キー発行を行います。外部公開する場合は、強い管理者パスワード、ネットワーク制限、TLS、バックアップ、監査ログ運用を必ず検討してください。

## ライセンス

MIT
