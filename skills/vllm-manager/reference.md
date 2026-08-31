# vLLM Manager — API reference

## Ports (docker-compose defaults)

| Service | Port | Role |
|---------|------|------|
| frontend | 13000 | Web UI (LLM / Embedding / Reranker tabs) |
| backend | 18000 | Management API + OpenAI-compatible `/v1/*` proxy |
| litellm-gateway | 14000 | LiteLLM inference gateway for external apps |

## Task types

| task_type | Start behavior | Smoke test | Proxy routing |
|-----------|----------------|------------|---------------|
| `chat` | Normal generate | `POST /v1/chat/completions` | `chat/completions`, `completions`, `messages` |
| `embedding` | `--runner pooling`, new instance | `POST /v1/embeddings` | `embeddings` |
| `rerank` | `--runner pooling`, new instance | `POST /v1/score` (fallback `/score`, `/v1/rerank`, `/rerank`) | `score`, `rerank`, `v1/score`, `v1/rerank`, `v2/rerank` |

CrossEncoder (sentence-transformers) is **not** started by Manager.

## Authentication

### Admin API authentication

**Question:** 「管理者APIの認証が必要」/ 401 on `/api/start` — what token?

**Answer:** `Authorization: Bearer vlmk_...` (PAT) or session JWT from **admin** login. **Not** `sk-`.

| Endpoint group | Required role | Token |
|----------------|---------------|-------|
| `/api/start`, `/api/stop`, `/api/instances/*`, `/api/model-downloads*`, `/api/models` POST/DELETE, `/api/users/*` | **admin** | PAT or session JWT (admin) |
| `/api/status`, `/api/models` GET, `/api/chat/*` | any logged-in user | PAT or session JWT |
| `:14000/v1/*` inference | LiteLLM key holder | `sk-` only |
| `:18000/v1/*` proxy | typically open (path-routed to running instance) | no PAT |

PAT **inherits the role** of the creating user. Non-admin PAT → 403 on start.

#### Method 1 — CLI (recommended)

Reads `VLLM_MANAGER_ADMIN_USER` / `VLLM_MANAGER_ADMIN_PASSWORD` from the repo `.env` (auto-discovered by walking up from the script, or `VLLM_MANAGER_ENV=/path/to/.env`).

```bash
# From the vllm-manager checkout (or with VLLM_MANAGER_ENV set):
scripts/vllm-cli.sh token create --name my-automation
# optional overrides: --username / --password
```

#### Method 2 — Web UI

Login `:13000` as admin → マイページ → 永続APIトークン（PAT）

#### Method 3 — Session JWT (short-lived)

```bash
TOKEN=$(curl -sS "$VLLM_MANAGER_URL/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"<password>"}' | jq -r .token)
```

Lost after `docker compose restart backend`. Prefer PAT for automation.

### LiteLLM key — inference (`sk-...`)

Works on `http://<host>:14000/v1/*` (chat, embeddings; rerank if gateway routes it through backend).

**Do not use PAT (`vlmk_`) on `:14000`.** Obtain sk- with a logged-in PAT.

**Model allow-list:** LiteLLM keys are scoped to models. For first setup / agents use **ensure** (all = `*`):

```bash
scripts/vllm-cli.sh inference-key ensure
# Re-running with same models/alias reuses the key (no duplicates).
# Force a new key: inference-key ensure --force
# Restrict: inference-key ensure --models 'vllm-local,jinaai/jina-embeddings-v3'
```

PAT (`token create`) has **no** model list — it only authenticates management `/api/*`.

| CLI | Auth | Notes |
|-----|------|-------|
| `inference-key ensure [--models] [--force]` | PAT | `POST /api/auth/me/api-keys/ensure` (preferred) |
| `inference-key create [--models …] [--save]` | PAT | Always mints a new key |
| `inference-key list` | PAT | `GET /api/auth/me/api-keys` |
| `inference-key delete --key <id>` | **admin** PAT | `POST /api/litellm/keys/delete` |
| `inference-key show` | none | Reports whether a saved/env sk- is found |

Config pack: [`skills/vllm-manager/.env.example`](.env.example) → copy to `.env`.  
Also available in UI (マイページ / ユーザ管理) and `POST /api/auth/me/api-keys`.

## Management API (selected)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/status` | any | Primary instance status |
| GET | `/api/servers` | admin | Running `vllm serve` processes |
| GET | `/api/instances` | admin | Managed instances (`task_type` included) |
| POST | `/api/start` | admin | Start vLLM (`model_id`, `task_type`, `context_length`, `create_new_instance`, …) |
| POST | `/api/stop` | admin | Stop default instance |
| POST | `/api/instances/stop` | admin | Body `{ "instance_id": "..." }` |
| POST | `/api/instances/{id}/smoke-test` | admin | chat / embeddings / score by task_type |
| GET | `/api/models` | any | Catalog |
| POST | `/api/models` | admin | Register / upsert (`task_type`: chat\|embedding\|rerank) |
| DELETE | `/api/models/{id}` | admin | Remove catalog + cache |
| POST | `/api/model-downloads` | admin | Start download |
| GET | `/api/model-downloads` | admin | List jobs (also reconciles orphans) |
| POST | `/api/model-downloads/resume` | admin | Resume from HF cache |
| POST | `/api/model-downloads/cancel` | admin | Cancel active jobs |
| GET | `/api/storage` | admin | Drive usage summary (NVMe / SATA, instant) |
| GET | `/api/storage/usage` | admin | Per-model (HF caches), Ollama, per-job sizes; `?refresh=true` rescans |
| GET | `/api/storage/breakdown` | admin | `?path=/home&top=60` directory sizes (du, cached; minutes on cold scan) |
| POST | `/api/training/datasets` | admin | Upload JSONL dataset (multipart `file=`) |
| GET | `/api/training/datasets` | admin | List uploaded datasets |
| POST | `/api/training/jobs` | admin | Submit LoRA SFT / DPO / GRPO job (1 concurrent max) |
| GET | `/api/training/jobs` / `/{id}` / `/{id}/log` | admin | List / detail (`?log_tail=N`) / log |
| POST | `/api/training/jobs/{id}/cancel` | admin | Cancel running job |
| POST | `/api/training/jobs/{id}/deploy` | admin | Hot-load adapter: `{port, lora_name}` (target needs `enable_lora`) |

Training details (dataset formats, hyperparams, GRPO rewards): repo `docs/training-api.md`.

### `POST /api/start` body (important fields)

```json
{
  "model_id": "BAAI/bge-reranker-v2-m3",
  "task_type": "rerank",
  "context_length": 8192,
  "create_new_instance": true,
  "instance_name": "rerank-bge",
  "download_model": false
}
```

If `task_type` omitted, catalog value is used. `embedding` / `rerank` force `create_new_instance` and default context 8192 when unset.

Additional chat fields: `gpu_devices` (`"1"` etc.; prefer explicit on multi-GPU hosts — `"all"` may grab a full GPU 0), `enable_lora` + optional `max_lora_rank` (allow runtime `/v1/load_lora_adapter` for training deploy).

## Inference paths (backend proxy `:18000`)

| Client path | Preferred instance |
|-------------|--------------------|
| `/v1/chat/completions` | `task_type=chat` |
| `/v1/embeddings` | `task_type=embedding` |
| `/v1/score`, `/score` | `task_type=rerank` |
| `/v1/rerank`, `/rerank`, `/v2/rerank` | `task_type=rerank` |

503 if no matching running instance.

**Model-name routing + round-robin:** when the request `model` matches a running instance's model id (exact, or basename match), that instance is used. If the **same model runs on multiple instances** (e.g. GPU0 and GPU1), requests round-robin across them. Aliases (`vllm-local` etc.) skip name matching and go to the first managed chat instance.

## Inference API (LiteLLM `:14000`)

| Method | Path | Auth |
|--------|------|------|
| GET | `/v1/models` | Bearer `sk-` |
| POST | `/v1/chat/completions` | Bearer `sk-` — **use `"stream": true`** |
| POST | `/v1/embeddings` | Bearer `sk-` |

Aliases: `vllm-local`, `claude-vllm-local` (→ first managed chat instance). The `*` wildcard now **passes the requested model name through** to the backend (`litellm_params.model: openai/*`), so specifying a real model id (e.g. `Qwen/Qwen3.8-27B-FP8`) routes to that instance — and load-balances round-robin when the same model runs on multiple GPUs.

Chat via `:14000` is force-streamed by the Manager (`PROXY_FORCE_STREAM` / `force_stream`, 504 avoidance). Clients that send `stream: false` get SSE back and LiteLLM returns 500 (`Empty or invalid response`). Prefer `stream: true`, or call backend `:18000/v1/chat/completions` for non-stream JSON.

## CLI (`scripts/vllm-cli.sh`)

| Command | Auth | Notes |
|---------|------|-------|
| `status` | none | Public status |
| `servers` | PAT | Running processes |
| `instances` / `instances stop --id` | PAT | Managed instances |
| `start <id>` | PAT | `--task-type chat\|embedding\|rerank`, `--context-length`, `--instance-name`, `--no-download` |
| `stop` / `restart` | PAT | Default instance |
| `smoke-test <instance_id>` | PAT | |
| `models list [--task-type t]` | none | Filter catalog |
| `models register <id> --task-type …` | PAT | Upsert catalog |
| `models download` / `downloads` / `resume` / `cancel` | PAT | HF jobs |
| `token create\|list\|revoke` | login / PAT | Management PAT (`vlmk_`) |
| `inference-key create\|list\|show` | PAT | LiteLLM sk-; `--save` → `litellm-key` file |
| `inference-key delete --key` | admin PAT | Deletes via LiteLLM `/key/delete` |
| `storage [overview\|usage\|breakdown <path>]` | admin PAT | Drive / per-model / directory sizes |
| `training jobs\|job\|log\|submit\|cancel\|datasets\|upload\|deploy` | admin PAT | LoRA SFT / DPO / GRPO jobs |

Environment: `VLLM_MANAGER_URL`, `VLLM_MANAGER_TOKEN`, `VLLM_MANAGER_ENV` (path to `.env`), `VLLM_MANAGER_CONFIG`, `LITELLM_API_KEY` / `VLLM_MANAGER_SK_KEY`.  
`token create` credentials: flags → `VLLM_MANAGER_USERNAME`/`PASSWORD` → `.env` `VLLM_MANAGER_ADMIN_USER`/`PASSWORD`.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| 401 on `/api/*` | Missing/invalid PAT; re-run `token create` |
| 403 Admin role required | Non-admin PAT — recreate with admin |
| 401 with `sk-` on `/api/*` | Wrong token type |
| 401 on `:14000/v1/*` | Missing sk-; run `inference-key create --save` |
| LiteLLM chat 500 / Empty or invalid response | Use `"stream": true` (Manager force-streams LiteLLM chat). Or hit `:18000` for non-stream |
| 503 on `/v1/embeddings` | No embedding instance running |
| 503 on `/v1/score` | No rerank instance running |
| Rerank start fails for Ruri CrossEncoder | Expected — use vLLM-native models (e.g. BGE) or Phase 2 |
| Model shows under wrong tab | Change 用途 in UI or `models register --task-type` |
| Download stuck as running | Backend reconciles orphans; try `models resume` after re-login |

Full docs in repo: `docs/api-requests.md`, `docs/security-and-operations.md`, `docs/teacher-guide.md`.
