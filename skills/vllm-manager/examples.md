# vLLM Manager — examples for agents

Replace `<HOST>`, tokens, and paths for your environment.

```bash
export VLLM_MANAGER_URL=http://<host>:18000
export LITELLM_URL=http://<host>:14000
CLI=./.cursor/skills/vllm-manager/scripts/vllm-cli.sh   # or skills/.../vllm-cli.sh
```

## Install skill in another project

```bash
cp -r /path/to/vllm-manager/skills/vllm-manager ~/.cursor/skills/
# or project-local under .cursor/skills/
```

## Example 0: Admin PAT + inference sk- (required setup)

Uses [`skills/vllm-manager/.env`](.env) (`VLLM_MANAGER_ADMIN_USER` / `PASSWORD`, `VLLM_MANAGER_INFERENCE_MODELS=*`).

```bash
$CLI token create --name automation                 # PAT — no model list
$CLI inference-key ensure                           # sk- models=* get-or-create
$CLI inference-key ensure                           # second call → reused: true
# Restrict / refresh:
# $CLI inference-key ensure --models 'vllm-local' --force
$CLI inference-key show
$CLI status
$CLI instances
```

PAT → `~/.config/vllm-manager/token`. Inference sk- → `~/.config/vllm-manager/litellm-key`.

**Still 403 on start?** Non-admin credentials in skill `.env`, or pass `--username`/`--password`.

Load sk- for curl:

```bash
SK="${LITELLM_API_KEY:-${VLLM_MANAGER_SK_KEY:-$(cat ~/.config/vllm-manager/litellm-key)}}"
```

## Example 1: Chat LLM — start + smoke + inference

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct
$CLI models download "$MODEL"
$CLI start "$MODEL" --context-length 32768 --no-download
INSTANCE=$($CLI instances | jq -r '[.[] | select(.running==true)][0].instance_id')
$CLI smoke-test "$INSTANCE"

SK="${LITELLM_API_KEY:-${VLLM_MANAGER_SK_KEY:-$(cat ~/.config/vllm-manager/litellm-key)}}"
# LiteLLM chat must use stream:true (backend force-streams these requests for 504 avoidance)
curl -sS -N "$LITELLM_URL/v1/chat/completions" \
  -H "Authorization: Bearer $SK" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"vllm-local\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":16,\"stream\":true}"
```

## Example 2: Embedding — register, start, call `/v1/embeddings`

```bash
EMB=jinaai/jina-embeddings-v3   # or sbintuitions/sarashina-embedding-v2-1b
$CLI models register "$EMB" --task-type embedding --context-length 8192 --size embed
$CLI models download "$EMB"
$CLI start "$EMB" --task-type embedding --context-length 8192 --no-download \
  --instance-name embed-main
$CLI instances
$CLI smoke-test "$($CLI instances | jq -r '.[] | select(.task_type=="embedding" and .running==true) | .instance_id' | head -1)"

curl -sS "$VLLM_MANAGER_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$EMB\",\"input\":\"対角化とは何か\"}" | jq '.data[0].embedding | length'
```

Via LiteLLM (needs sk- from Example 0):

```bash
SK="${LITELLM_API_KEY:-${VLLM_MANAGER_SK_KEY:-$(cat ~/.config/vllm-manager/litellm-key)}}"
curl -sS "$LITELLM_URL/v1/embeddings" \
  -H "Authorization: Bearer $SK" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$EMB\",\"input\":\"hello\"}"
```

## Example 3: Reranker (vLLM-native) — start + score + rerank

```bash
RR=BAAI/bge-reranker-v2-m3
$CLI models register "$RR" --task-type rerank --context-length 8192 --size rerank
$CLI models download "$RR"
$CLI start "$RR" --task-type rerank --context-length 8192 --no-download \
  --instance-name rerank-bge
$CLI smoke-test "$($CLI instances | jq -r '.[] | select(.task_type=="rerank" and .running==true) | .instance_id' | head -1)"

# Pairwise score
curl -sS "$VLLM_MANAGER_URL/v1/score" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$RR\",\"text_1\":\"対角化とは\",\"text_2\":\"行列を対角行列に変換する操作\"}" | jq .

# Document list rerank
curl -sS "$VLLM_MANAGER_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$RR\",\"query\":\"対角化とは\",\"documents\":[\"無関係な文\",\"行列を対角化する\"]}" | jq .
```

**Do not** expect Manager start to work for CrossEncoder-only models (e.g. many `hotchpotch/*-reranker*`, `cl-nagoya/ruri-v3-reranker-*`). Catalog/UI registration as `rerank` is fine for organization; runtime is Phase 2.

## Example 4: HTTP-only (no CLI) — start embedding / rerank

```bash
# PAT required
AUTH="Authorization: Bearer $VLLM_MANAGER_TOKEN"

curl -sS -X POST "$VLLM_MANAGER_URL/api/start" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "model_id": "BAAI/bge-reranker-v2-m3",
    "task_type": "rerank",
    "context_length": 8192,
    "create_new_instance": true,
    "download_model": false
  }' | jq .
```

## Example 5: Download resume / cancel

```bash
$CLI models downloads
$CLI models resume sbintuitions/sarashina-embedding-v2-1b
$CLI models cancel org/some-model
```

## Example 6: Fix misclassified task_type

```bash
# Was registered as embedding before rerank existed
$CLI models register BAAI/bge-reranker-v2-m3 --task-type rerank --context-length 8192
$CLI models list --task-type rerank
```

Or in UI: モデル管理 → 用途セレクト → Reranker.

## Example 7: Stop embedding/rerank without killing chat

```bash
$CLI instances
$CLI instances stop --id inst-bge-reranker-v2-m3-xxxxxxxx
# chat instance (if any) keeps running
```

## Example 8: Python — ensure embedding then embed

```python
import os, subprocess, httpx

CLI = os.path.expanduser("~/.cursor/skills/vllm-manager/scripts/vllm-cli.sh")
VLLM = os.environ["VLLM_MANAGER_URL"]
MODEL = "jinaai/jina-embeddings-v3"

subprocess.run([CLI, "start", MODEL, "--task-type", "embedding",
                "--context-length", "8192", "--no-download"], check=True)

r = httpx.post(
    f"{VLLM}/v1/embeddings",
    json={"model": MODEL, "input": "hello"},
    timeout=120,
)
r.raise_for_status()
print("dim", len(r.json()["data"][0]["embedding"]))
```

## Example 9: CI — download + start rerank + smoke

```bash
export VLLM_MANAGER_TOKEN=vlmk_...   # secrets
MODEL=BAAI/bge-reranker-v2-m3
$CLI models download "$MODEL"
$CLI start "$MODEL" --task-type rerank --context-length 8192 --no-download
ID=$($CLI instances | jq -r --arg m "$MODEL" \
  '.[] | select(.model==$m and .running==true) | .instance_id' | head -1)
$CLI smoke-test "$ID"
```

## Example 10: LoRA SFT → deploy → inference (all HTTP)

```bash
# dataset: JSONL of {"messages":[{"role":"user",...},{"role":"assistant",...}]}
$CLI training upload sft_data.jsonl
$CLI training submit --method sft --base-model Qwen/Qwen3.5-9B-Instruct \
  --dataset sft_data.jsonl --gpu 1 \
  --json '{"hyperparams":{"epochs":2,"lora_r":16,"max_seq_len":2048}}'
$CLI training jobs                       # wait for status=completed
JOB=<job_id>

# vLLM with runtime-LoRA enabled, then hot-load the adapter
$CLI start Qwen/Qwen3.5-9B-Instruct --json '{"enable_lora":true,"gpu_devices":"1"}'
PORT=$($CLI instances | jq -r '.[] | select(.running==true and (.model|contains("Qwen3.5-9B"))) | .vllm_port' | head -1)
$CLI training deploy "$JOB" --port "$PORT" --name my-adapter

curl -sS "$VLLM_MANAGER_URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"my-adapter","messages":[{"role":"user","content":"hi"}]}'
```

## Example 11: GRPO (RL) with exact-match reward

```bash
# dataset rows: {"prompt": "...", "answer": "42"}
$CLI training upload math.jsonl
$CLI training submit --method grpo --base-model Qwen/Qwen3.5-9B-Instruct \
  --dataset math.jsonl --gpu 1 \
  --json '{"reward":{"type":"exact_match"},"hyperparams":{"num_generations":4}}'
# remote reward server instead: {"reward":{"type":"remote","url":"http://host:9000/reward"}}
# (POST {prompts, completions} -> {"rewards": [...]})
```

## Example 12: Storage — find what fills the SSDs

```bash
$CLI storage                             # NVMe vs SATA: used/free
$CLI storage breakdown /home             # biggest dirs on the SATA SSD
$CLI storage breakdown /home/kaihara/workspace/python_env   # drill deeper
$CLI storage usage                       # HF caches per model / Ollama / training jobs
```

## Example 13: Minimal-VRAM startup (share a GPU without hogging it)

```bash
# chat: sized from context_length x max_num_seqs (exact --kv-cache-memory bytes)
$CLI start Qwen/Qwen2.5-7B-Instruct --context-length 8192 --no-download \
  --json '{"gpu_devices":"1","gpu_memory_mode":"minimal","max_num_seqs":4}'
# -> steps report the computed KV cache GiB + utilization ceiling used

# embedding/rerank: sized from model weight bytes only (no KV cache needed)
$CLI models register jinaai/jina-embeddings-v3 --task-type embedding --context-length 8192
$CLI start jinaai/jina-embeddings-v3 --task-type embedding --context-length 8192 --no-download \
  --instance-name embed-minimal \
  --json '{"gpu_devices":"1","gpu_memory_mode":"minimal"}'
```

Falls back to `auto` (with a reason in `steps`) if the model's HF config/file
list can't be fetched — e.g. gated repo without `HF_TOKEN`, or offline.

## Token hygiene

- Never commit `vlmk_` or `sk-` keys.
- PAT: `~/.config/vllm-manager/token` (mode 600).
- Rotate via UI or `token revoke` + `token create`.
