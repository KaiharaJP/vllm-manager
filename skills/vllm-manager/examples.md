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

## Example 0: Admin PAT (required for start/download)

```bash
$CLI token create --name automation --username admin --password '<admin-password>'
$CLI status
$CLI instances
```

**Still 403?** Token was issued by a non-admin user. Re-create with admin.

## Example 1: Chat LLM — start + smoke + inference

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct
$CLI models download "$MODEL"
$CLI start "$MODEL" --context-length 32768 --no-download
INSTANCE=$($CLI instances | jq -r '[.[] | select(.running==true)][0].instance_id')
$CLI smoke-test "$INSTANCE"

curl -sS "$LITELLM_URL/v1/chat/completions" \
  -H "Authorization: Bearer $SK" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"vllm-local\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":16}"
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

Via LiteLLM (needs `sk-`):

```bash
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

## Token hygiene

- Never commit `vlmk_` or `sk-` keys.
- PAT: `~/.config/vllm-manager/token` (mode 600).
- Rotate via UI or `token revoke` + `token create`.
