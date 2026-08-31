---
name: vllm-manager
description: >-
  Operate vLLM Manager (start/stop chat, embedding, rerank vLLM; download models;
  smoke-test; inference via LiteLLM or backend /v1; LoRA/DPO/GRPO training jobs;
  storage usage inspection). Use when the user mentions vllm-manager,
  embedding/reranker startup, model download automation, admin API auth, PAT
  tokens (vlmk_), LiteLLM sk- keys, vllm-cli, LoRA fine-tuning / RL on this
  server, disk/SSD usage breakdown, or self-hosted LLM on :18000/:14000.
---

# vLLM Manager

Self-hosted LLM ops stack: FastAPI backend (`:18000`), LiteLLM gateway (`:14000`), web UI (`:13000`).

Supports three **task types** (vLLM processes):

| task_type | UI label | vLLM flag | Inference paths |
|-----------|----------|-----------|-----------------|
| `chat` | LLM | generate (default) | `/v1/chat/completions` |
| `embedding` | Embedding | `--runner pooling` | `/v1/embeddings` |
| `rerank` | Reranker | `--runner pooling` | `/v1/score`, `/v1/rerank`, `/score`, `/rerank` |

**Out of scope:** CrossEncoder / sentence-transformers (Ruri CrossEncoder 等). Register as `rerank` for catalog UI only; do not expect Manager start to work until Phase 2.

## Three auth types (do not mix)

| Token | Prefix | Use for |
|-------|--------|---------|
| Session JWT | (opaque) | Browser login; lost on backend restart |
| PAT | `vlmk_` | **Management API** `/api/*` — start/stop, downloads, CLI |
| LiteLLM key | `sk-` | **Inference** via `:14000/v1/*` |

PAT cannot call LiteLLM inference. `sk-` cannot call `/api/start`. Chat UI uses session JWT only.

### Which “token” and which models?

| Step | Token | Model selection |
|------|-------|-----------------|
| `token create` | PAT `vlmk_` | **None** — management only (start/stop/download) |
| `inference-key create` | LiteLLM `sk-` | **Yes** — allow-list for `:14000/v1/*` (prefer **ensure**) |

For agents and first setup, issue an sk- with **all models** via **ensure** (get-or-create; no duplicate keys):

- Default: `models=["*"]`, alias `vllm-cli` (`VLLM_MANAGER_INFERENCE_MODELS=*` in skill `.env`)
- Same models + alias → returns the existing key (server-stored)
- Restrict later: `--models vllm-local,...` or `--force` to mint a new one

## Admin API authentication — how to obtain

When the user sees **401 Unauthorized**, **403 Admin role required**, or **「管理者APIの認証が必要」**:

1. They need a PAT (`vlmk_...`) or session JWT — **NOT** `sk-`.
2. Start/stop/download require **admin** role.

```bash
export VLLM_MANAGER_URL=http://<host>:18000   # optional; else skills .env / BACKEND_PORT
CLI=skills/vllm-manager/scripts/vllm-cli.sh   # adjust path

# Credentials + defaults from skills/vllm-manager/.env
$CLI token create --name automation
$CLI inference-key ensure                 # models=* get-or-create (no duplicates)
$CLI status
```

If the skill is installed outside the repo, keep using `skills/vllm-manager/.env` or:

```bash
export VLLM_MANAGER_ENV=/path/to/skills/vllm-manager/.env
```

Full auth guide: [reference.md](reference.md#admin-api-authentication).

## Quick setup (other projects)

```bash
cp -r /path/to/vllm-manager/skills/vllm-manager ~/.cursor/skills/
# Edit ~/.cursor/skills/vllm-manager/.env (from .env.example)
CLI=~/.cursor/skills/vllm-manager/scripts/vllm-cli.sh
$CLI token create --name automation
$CLI inference-key ensure                 # all models (*); reuses if already issued
```

Requires `curl` and `jq`. PAT → `~/.config/vllm-manager/token`; sk- → `~/.config/vllm-manager/litellm-key`.

## Common workflows

### A. Chat LLM

```bash
CLI=skills/vllm-manager/scripts/vllm-cli.sh
$CLI models download Qwen/Qwen2.5-7B-Instruct
$CLI start Qwen/Qwen2.5-7B-Instruct --context-length 32768
$CLI instances
$CLI smoke-test <instance_id>
```

Inference (LiteLLM) — use sk- from `inference-key create --save` (or `LITELLM_API_KEY`).
**Always set `"stream": true` for chat via `:14000`** (Manager force-streams LiteLLM chat to avoid 504; non-stream clients get parse errors).

```bash
export LITELLM_URL="${LITELLM_URL:-http://localhost:14000}"
SK_KEY="${LITELLM_API_KEY:-${VLLM_MANAGER_SK_KEY:-$(cat ~/.config/vllm-manager/litellm-key)}}"
curl -sS -N "$LITELLM_URL/v1/chat/completions" \
  -H "Authorization: Bearer $SK_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"vllm-local","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"stream":true}'
```

### B. Embedding

```bash
$CLI models register jinaai/jina-embeddings-v3 --task-type embedding --context-length 8192
$CLI models download jinaai/jina-embeddings-v3
$CLI start jinaai/jina-embeddings-v3 --task-type embedding --context-length 8192 --no-download \
  --instance-name embed-jina
$CLI smoke-test <instance_id>
```

Inference (prefer backend `:18000` or LiteLLM gateway `:14000` — both accept any running model id):

```bash
# via :14000 (gateway passes embeddings/rerank to Manager; no fixed LiteLLM catalog entry)
curl -sS "$LITELLM_URL/v1/embeddings" \
  -H "Authorization: Bearer $SK_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"jinaai/jina-embeddings-v3","input":"対角化とは"}'
```

### C. Reranker (vLLM-native only)

```bash
$CLI models register BAAI/bge-reranker-v2-m3 --task-type rerank --context-length 8192
$CLI models download BAAI/bge-reranker-v2-m3
$CLI start BAAI/bge-reranker-v2-m3 --task-type rerank --context-length 8192 --no-download \
  --instance-name rerank-bge
$CLI smoke-test <instance_id>
```

Score / rerank:

```bash
# Pair score
curl -sS "$VLLM_MANAGER_URL/v1/score" \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-reranker-v2-m3","text_1":"対角化とは","text_2":"行列を対角化する操作"}'

# Cohere-style rerank
curl -sS "$VLLM_MANAGER_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-reranker-v2-m3","query":"対角化とは","documents":["doc A","doc B"]}'
```

Catalog already has `task_type=rerank` → you may omit `--task-type` on start (server reads catalog).

### D. Downloads (resume / cancel)

```bash
$CLI models downloads
$CLI models resume org/model-name
$CLI models cancel org/model-name
```

### E. Multi-instance ops

```bash
$CLI servers          # all vllm serve processes
$CLI instances        # managed instances + task_type
$CLI instances stop --id <instance_id>
$CLI stop             # stop default instance only
```

VRAM tip: do not keep a large chat model and embedding/rerank on the same GPU unless capacity allows. Stop chat first if needed.

### F. Model-name routing / multi-GPU load balancing

`:14000` chat passes the real model id through to the backend. Prefer a concrete
model id over `vllm-local` when multiple chat models run:

```bash
curl -sS -N "$LITELLM_URL/v1/chat/completions" \
  -H "Authorization: Bearer $SK_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Same model started on multiple GPUs (same model_id, different `gpu_devices`)
→ requests **round-robin** across instances automatically. On multi-GPU hosts
always pass explicit `--json '{"gpu_devices":"1"}'` (or UI setting) at start;
`"all"` tends to grab GPU 0 and fail when it is occupied.

### G. Training jobs (LoRA SFT / DPO / GRPO) — HTTP only

```bash
$CLI training upload my_data.jsonl
$CLI training submit --method sft --base-model Qwen/Qwen3.5-9B-Instruct \
  --dataset my_data.jsonl --gpu 1
$CLI training jobs                      # status/progress/loss
$CLI training log <job_id> --tail 50
# Serve the result without restart:
$CLI start Qwen/Qwen3.5-9B-Instruct --json '{"enable_lora":true,"gpu_devices":"1"}'
$CLI training deploy <job_id> --port <vllm_port> --name my-adapter
# → inference with "model": "my-adapter"
```

Rules: 1 job at a time; `--gpu` explicit (checked against free VRAM);
base = **non-quantized** model (not `*-FP8`) with 4bit QLoRA. GRPO needs
`--json '{"reward":{"type":"exact_match"}}'` (or `contains` / `remote` with url).
Dataset formats + hyperparams: repo `docs/training-api.md`.

### H. Storage usage (SSD 内訳)

```bash
$CLI storage                            # both drives: used/free
$CLI storage usage                      # HF caches per model, Ollama, training jobs
$CLI storage breakdown /home            # directory sizes; drill by repeating with deeper path
```

UI: ストレージ tab (admin). Cold scans take minutes; results are cached
(`--refresh` to rescan). Sizes come from a read-only host mount; nothing can be
deleted through this API.

### I. Fix wrong task_type

UI: モデル管理 → 各カードの「用途」セレクト。  
CLI: re-register with the correct `--task-type` (upsert).

```bash
$CLI models register BAAI/bge-reranker-v2-m3 --task-type rerank --context-length 8192
```

Names containing `rerank` that were saved as embedding are auto-migrated on backend startup.

## Agent checklist

1. **401/403 on admin API** → obtain admin PAT first (`token create`).
2. Confirm `VLLM_MANAGER_URL` and token type (`vlmk_` vs `sk-`).
3. **Ops** → admin PAT + `vllm-cli.sh` or `/api/*`.
4. **Inference via `:14000`** → `inference-key ensure` (default models=`*` all; reuses existing).
5. Pick **task_type**: chat / embedding / rerank (never start CrossEncoder via Manager).
6. After start → `instances` + `smoke-test` before assuming ready.
7. Inference: chat via `:14000` with sk- and **`stream: true`** (required); embeddings via `:14000` with sk-, or backend `:18000/v1/*` path routing by subpath. Non-stream chat → use `:18000` directly.
8. Prefer a **concrete model id** over `vllm-local` when multiple chat models run; same model on 2 GPUs load-balances automatically.
9. On multi-GPU hosts pass explicit `gpu_devices` at start (`"all"` grabs GPU 0 and may OOM).
10. Training / storage ops → admin PAT + `training ...` / `storage ...` CLI commands (or `/api/training/*`, `/api/storage*`).
11. Do not commit tokens (PAT file and litellm-key are separate).

## Defaults

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_MANAGER_URL` | `http://localhost:18000` | Backend API |
| `NEXT_PUBLIC_LITELLM_URL` / `LITELLM_URL` | `http://localhost:14000` | Inference gateway |
| `VLLM_MANAGER_TOKEN` | `~/.config/vllm-manager/token` | PAT override |
| `LITELLM_API_KEY` / `VLLM_MANAGER_SK_KEY` | `~/.config/vllm-manager/litellm-key` | Inference sk- |
| `VLLM_MANAGER_INFERENCE_MODELS` | `*` | Default allow-list for `inference-key create` |
| Skill `.env` | `skills/vllm-manager/.env` | Admin login + URL + models default |

## More detail

- API & ports: [reference.md](reference.md)
- Copy-paste examples: [examples.md](examples.md)
