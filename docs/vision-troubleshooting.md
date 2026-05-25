# Vision（画像入力）トラブルシューティング

Hermes / LiteLLM 経由で「画像が読めない」「Empty or invalid response」が出るときの確認手順です。

---

## 1. vLLM が Vision を有効にしているか

起動ログ（`/app/data/vllm.log`）で次を確認します。

```bash
docker exec vllm-manager-backend grep -i "Encoder cache" /app/data/vllm.log | tail -5
```

| ログ | 意味 |
|------|------|
| `Encoder cache will be initialized ... N image items` | Vision エンコーダ **有効** |
| `explicitly disabled via limit_mm_per_prompt` | **`limit_mm_per_prompt` で image:0 等** — Vision 無効 |

起動コマンドに `--limit-mm-per-prompt '{"image":1}'` が付いているか:

```bash
docker exec vllm-manager-backend pgrep -af "vllm serve"
```

UI の **Vision / マルチモーダル** セクション、または `config.json` の `limit_mm_per_prompt` を確認してください。

---

## 2. リクエストに画像が届いているか

`request_history.jsonl`（管理 UI のリクエスト履歴と同じデータ）:

```bash
docker exec vllm-manager-backend python3 -c "
import json
from pathlib import Path
for line in Path('/app/data/request_history.jsonl').open():
    rec = json.loads(line)
    s = json.dumps(rec.get('messages') or [], ensure_ascii=False)
    if 'image_url' in s or 'input_image' in s:
        print(rec.get('model'), rec.get('prompt_tokens'), rec.get('status'), rec.get('endpoint'))
"
```

- `image_url` / `input_image` が無い → **クライアントまたは LiteLLM 側**で画像が落ちている
- `prompt_tokens` がテキストのみより明らかに少ない → 画像がトークン化されていない可能性

---

## 3. 切り分けテスト（A/B）

### A: vLLM 直叩き（プロキシ・LiteLLM なし）

```bash
# コンテナ内またはホストから vLLM ポートへ（例: 8011）
curl -s http://127.0.0.1:8011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.6-27B-FP8",
    "max_tokens": 64,
    "stream": false,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What color is this 1x1 pixel? One word."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}}
      ]
    }]
  }'
```

- 画像の色に言及する → **モデル・vLLM は正常** → プロキシ / Hermes 側を疑う
- 無視する → **MTP を off**、コンテキストを短く（8192）して再試行

### B: backend 経由（LiteLLM ヘッダなし）

`X-Vllm-Manager-Source: litellm` を付けない POST で `stream` の有無を比較します。

### C: `PROXY_FORCE_STREAM=false`

`.env` または `docker-compose.yml` で:

```env
PROXY_FORCE_STREAM=false
```

`docker compose up -d backend` 後、Hermes で再試行。  
「Empty or invalid response」が消える場合、**ストリーム強制とクライアントの相性**が原因です。

**本リポジトリの既定修正:** 画像付き `chat/completions` では `force_stream` を自動スキップします（`app/main.py`）。UI の「LiteLLM 経由でテキストのみのとき stream: true を強制」はテキスト専用の 504 対策として残ります。

---

## 4. 起動オプション（Qwen3.6 向け）

| オプション | 用途 |
|------------|------|
| `--limit-mm-per-prompt '{"image":1}'` | 1 リクエストあたりの画像上限（**0 で Vision 無効**） |
| `--mm-encoder-tp-mode data` | 高負荷・TP>1 時の Vision エンコーダ（任意） |
| `--mm-processor-cache-type shm` | 前処理キャッシュ（任意） |

詳細は [server-start-options.md](./server-start-options.md) を参照。

---

## 5. MTP と超長コンテキスト

- `speculative_config`（MTP）有効時は、Vision 単体テストで **一度 off** にして比較
- `context_length: 262144` + 巨大 base64 履歴は、画像トークンが埋もれて「見ていない」回答になりうる

---

## 6. `/v1/responses` 経路

backend は `input_image` → `image_url` に変換して `chat/completions` へブリッジします（`stream: false` 固定）。  
Hermes が `chat/completions` + LiteLLM 経由の場合は、上記 **force_stream スキップ** が効きます。
