# 学習ジョブ API（LoRA SFT / DPO / GRPO）

backend の HTTP API から LoRA 微調整・選好最適化（DPO）・強化学習系（GRPO）を投入できる。
学習は `scripts/train_job.py` がサブプロセスとして実行し、`/app/data/training/` 以下で管理される。

- 認証: 管理者トークン（`/api/auth/login` で取得）
- 同時実行: **1 ジョブまで**（GPU の取り合い防止）
- GPU: `gpu_devices` の**明示指定が必須**（`"all"` 不可）。投入時に空き VRAM をチェックする

## 全体フロー

```
①データセットをアップロード → ②ジョブ投入 → ③進捗ポーリング
→ ④enable_lora で vLLM 起動 → ⑤アダプタをホットロード → ⑥推論
```

## ① データセットのアップロード

```bash
TOKEN=$(curl -s -X POST http://localhost:18000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"***"}' | jq -r .access_token)

curl -X POST http://localhost:18000/api/training/datasets \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@my_data.jsonl"
```

### データセット形式（JSONL、1 行 1 サンプル）

| method | 必須フィールド |
|--------|---------------|
| `sft`  | `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` または `{"text": "..."}` |
| `dpo`  | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| `grpo` | `{"prompt": "..."}` ＋ reward.type に応じて `"answer"` / `"keywords"` |

## ② ジョブ投入

### SFT（QLoRA）

```bash
curl -X POST http://localhost:18000/api/training/jobs \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "method": "sft",
    "base_model": "Qwen/Qwen3.5-9B-Instruct",
    "dataset": "my_data.jsonl",
    "gpu_devices": "1",
    "quantization": "4bit",
    "hyperparams": {"epochs": 2, "lora_r": 16, "learning_rate": 2e-4, "max_seq_len": 2048}
  }'
```

### DPO（選好最適化）

```bash
curl -X POST http://localhost:18000/api/training/jobs \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "method": "dpo",
    "base_model": "Qwen/Qwen3.5-9B-Instruct",
    "dataset": "prefs.jsonl",
    "gpu_devices": "1",
    "hyperparams": {"dpo_beta": 0.1, "learning_rate": 5e-6}
  }'
```

### GRPO（強化学習系）

報酬は 3 方式:

- `exact_match`: データセットの `answer` と完全一致で 1.0
- `contains`: `keywords`（list）の含有率
- `remote`: 外部報酬サーバーへ `POST {prompts, completions}` → `{"rewards": [...]}`

```bash
curl -X POST http://localhost:18000/api/training/jobs \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "method": "grpo",
    "base_model": "Qwen/Qwen3.5-9B-Instruct",
    "dataset": "math_problems.jsonl",
    "gpu_devices": "1",
    "reward": {"type": "exact_match"},
    "hyperparams": {"num_generations": 4, "max_completion_length": 512}
  }'
```

## ③ 進捗確認・キャンセル

```bash
curl -s http://localhost:18000/api/training/jobs -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:18000/api/training/jobs/<job_id>?log_tail=50 -H "Authorization: Bearer $TOKEN"
curl -X POST http://localhost:18000/api/training/jobs/<job_id>/cancel -H "Authorization: Bearer $TOKEN"
```

`status.json` の `status`: `queued → running → completed | failed | cancelled`。
running 中は `step` / `total_steps` / `loss` / `progress` が更新される。

## ④⑤ 学習済みアダプタの配信

vLLM を `enable_lora: true` で起動しておく（`/api/start` に追加済み）:

```bash
curl -X POST http://localhost:18000/api/start \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen3.5-9B-Instruct", "gpu_devices": "1", "enable_lora": true}'
```

完了したジョブのアダプタをホットロード（vLLM 再起動不要）:

```bash
curl -X POST http://localhost:18000/api/training/jobs/<job_id>/deploy \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"port": 8013, "lora_name": "my-adapter"}'
```

以後 `"model": "my-adapter"` で推論できる:

```bash
curl -X POST http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-adapter", "messages": [{"role": "user", "content": "..."}]}'
```

## 制約・注意

- 学習と推論は VRAM を食い合う。27B クラスの QLoRA は片方の GPU をほぼ専有する
  （9B クラスなら 4bit で 20GB 前後）。`min_free_gb` で投入時の空きチェックを調整できる
- ベースモデルと同系のモデルにしかアダプタはロードできない
  （例: Qwen3.5-9B で学習した LoRA を Qwen3.6-27B には載せられない）
- FP8 量子化済みチェックポイント（*-FP8）はベースモデルに不向き。学習は非量子化版
  （例: `Qwen/Qwen3.5-9B-Instruct`）を 4bit ロードで行うこと
- ジョブの成果物は `/app/data/training/jobs/<job_id>/adapter/`（vllm-data ボリューム内）に残る
