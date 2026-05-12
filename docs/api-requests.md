# API リクエスト方法（Docker 構成）

このリポジトリを `docker compose` で起動したとき、LLM へアクセスする主な経路は次の2つです。

| 経路 | 用途の目安 |
|------|------------|
| **バックエンド（FastAPI）** | OpenAI 互換 `/v1/*` をそのまま利用。ブラウザや curl から直接試すとき |
| **LiteLLM プロキシ** | 外部ツール（Hermes Agent 等）から **1つの base URL** でまとめて呼ぶとき |

ポートは `.env` の `BACKEND_PORT` / `LITELLM_PORT` で変わります。以下では compose のデフォルト例として **`18000`（backend）** と **`14000`（LiteLLM）** を使います。実環境ではホスト名・ポートを読み替えてください。

---

## 前提

- `docker compose up -d` で **backend** と **LiteLLM** が起動していること。
- **vLLM** は管理 UI の「サーバー管理」から起動済みであること（未起動だと `/v1/models` が空になる、または 503 になることがあります）。
- `model` には、まず **`GET /v1/models`** で返る各エントリの `id`（例: `org/model-name`）を指定するのが確実です。

---

## 1. バックエンド経由（OpenAI 互換）

ベース URL:

```text
http://<ホスト>:<BACKEND_PORT>/v1
```

例（ホスト上からループバックで試す場合）:

```text
http://127.0.0.1:18000/v1
```

公開ドメイン経由の例:

```text
http://hinton.prv.kanazawa-it.ac.jp:18000/v1
```

### モデル一覧

```bash
curl -sS "http://127.0.0.1:18000/v1/models"
```

複数の vLLM を起動している場合、`data` に複数の `id` が並ぶことがあります（集約表示）。

### チャット補完（Chat Completions）

```bash
curl -sS "http://127.0.0.1:18000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "< /v1/models で返った id >",
    "messages": [
      {"role": "user", "content": "こんにちは。短く返答してください。"}
    ],
    "max_tokens": 256
  }'
```

`max_tokens` 等を省略した場合、サーバー管理で設定したデフォルト値が適用される場合があります。

### 補完（Completions / レガシー）

モデルが対応していれば `POST /v1/completions` も同様に利用できます。

---

## 2. LiteLLM プロキシ経由

ベース URL:

```text
http://<ホスト>:<LITELLM_PORT>/v1
```

例:

```text
http://127.0.0.1:14000/v1
```

### 認証

`.env` の `LITELLM_MASTER_KEY` が設定されている場合、リクエストに **Bearer トークン** が必要です。

```http
Authorization: Bearer <LITELLM_MASTER_KEY の値>
```

例（`.env.example` のデフォルト例）:

```bash
export LITELLM_KEY="sk-vllm-default-key"

curl -sS "http://127.0.0.1:14000/v1/models" \
  -H "Authorization: Bearer ${LITELLM_KEY}"
```

### 利用するモデル名（エイリアス）

`config/litellm_config.yaml` で定義された **`model_name`** を `model` に指定します。現在の例では汎用エイリアス **`vllm-local`** が定義されており、バックエンド側で起動中の vLLM へルーティングされます。

```bash
curl -sS "http://127.0.0.1:14000/v1/chat/completions" \
  -H "Authorization: Bearer ${LITELLM_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-local",
    "messages": [{"role": "user", "content": "1+1は？"}],
    "max_tokens": 64
  }'
```

設定を変更した場合は `config/litellm_config.yaml` を確認し、必要に応じて `docker compose restart litellm` で反映してください。

---

## 3. Claude Code / Anthropic 形式での利用（LiteLLM 経由）

Claude Code は `ANTHROPIC_BASE_URL` で LLM gateway を指定できます。  
この構成では LiteLLM が Anthropic Messages 形式（`/v1/messages`）を受け、上流の vLLM（OpenAI 互換）へ変換して転送します。

### 事前確認（Anthropic 形式の疎通）

```bash
curl -sS "http://127.0.0.1:14000/v1/messages" \
  -H "Authorization: Bearer ${LITELLM_KEY}" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-vllm-local",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "こんにちは。短く返答してください。"}]
  }'
```

### Claude Code 側の設定例

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:14000"
export ANTHROPIC_AUTH_TOKEN="${LITELLM_KEY}"

# 既定モデル（任意）
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-vllm-local"
```

この状態で `claude` を起動し、`/model claude-vllm-local` を指定すると LiteLLM 経由で利用できます。

---

## 4. フロントエンドの URL（参考）

UI から同じ API を叩く場合、ブラウザ用にビルド時の公開 URL が使われます。`.env` の例:

- `NEXT_PUBLIC_API_URL` … バックエンド（例: `http://...:18000`）
- `NEXT_PUBLIC_LITELLM_URL` … LiteLLM（例: `http://...:14000`）

---

## 5. トラブルシュート（よくあること）

| 現象 | 確認すること |
|------|----------------|
| `503` や接続エラー | vLLM が起動しているか、バックエンド・LiteLLM コンテナが `healthy` か |
| `401`（管理 API） | フロントの「サーバー起動」等はログイン必須。トークン切れなら再ログイン |
| `/v1/models` が期待と違う | ホストの **別ポート** に別アプリがバインドしていないか（過去、ホスト `:8007` とコンテナ内 vLLM の取り違え事例あり）。集約は **`<BACKEND_PORT>/v1/models`** を参照 |
| LiteLLM が通らない | `Authorization: Bearer` と `LITELLM_MASTER_KEY` の一致、`litellm` 再起動後の設定読み込み |
| Claude Code から繋がらない | `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` の設定、`/v1/messages` に curl で事前疎通できるか |

---

## 6. 関連ファイル

| ファイル | 内容 |
|----------|------|
| `docker-compose.yml` | ポートマップ、`backend` / `litellm` のサービス定義 |
| `.env` / `.env.example` | `BACKEND_PORT`, `LITELLM_PORT`, `LITELLM_MASTER_KEY`, `NEXT_PUBLIC_*` |
| `config/litellm_config.yaml` | LiteLLM の `model_list` とバックエンドへの `api_base` |
