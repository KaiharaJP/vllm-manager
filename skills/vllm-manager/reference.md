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

```bash
export VLLM_MANAGER_URL=http://<host>:18000
scripts/vllm-cli.sh token create --name my-automation --username admin --password '<password>'
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

## Inference paths (backend proxy `:18000`)

| Client path | Preferred instance |
|-------------|--------------------|
| `/v1/chat/completions` | `task_type=chat` |
| `/v1/embeddings` | `task_type=embedding` |
| `/v1/score`, `/score` | `task_type=rerank` |
| `/v1/rerank`, `/rerank`, `/v2/rerank` | `task_type=rerank` |

503 if no matching running instance.

## Inference API (LiteLLM `:14000`)

| Method | Path | Auth |
|--------|------|------|
| GET | `/v1/models` | Bearer `sk-` |
| POST | `/v1/chat/completions` | Bearer `sk-` |
| POST | `/v1/embeddings` | Bearer `sk-` |

Aliases: `vllm-local`, `claude-vllm-local`, `*` (wildcard → backend).

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
| `token create\|list\|revoke` | login / PAT | |

Environment: `VLLM_MANAGER_URL`, `VLLM_MANAGER_TOKEN`, `VLLM_MANAGER_USERNAME`, `VLLM_MANAGER_PASSWORD`, `VLLM_MANAGER_CONFIG`.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| 401 on `/api/*` | Missing/invalid PAT; re-run `token create` |
| 403 Admin role required | Non-admin PAT — recreate with admin |
| 401 with `sk-` on `/api/*` | Wrong token type |
| 503 on `/v1/embeddings` | No embedding instance running |
| 503 on `/v1/score` | No rerank instance running |
| Rerank start fails for Ruri CrossEncoder | Expected — use vLLM-native models (e.g. BGE) or Phase 2 |
| Model shows under wrong tab | Change 用途 in UI or `models register --task-type` |
| Download stuck as running | Backend reconciles orphans; try `models resume` after re-login |

Full docs in repo: `docs/api-requests.md`, `docs/security-and-operations.md`, `docs/teacher-guide.md`.
