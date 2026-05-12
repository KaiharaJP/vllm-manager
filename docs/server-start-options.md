# サーバー起動オプション詳細

`サーバー管理` タブで指定する値が、実際に何へ効くかをまとめたガイドです。  
「どれを触るべきか分からない」状態から始められるように、推奨初期値と失敗時の見方を先に書いています。

---

## まず最初の推奨設定

迷ったら次で開始し、必要になってから上げるのが安全です。

- `context_length`: `8192` または `32768`
- `max_num_seqs`: `1` 〜 `2`
- `gpu_memory_mode`: `auto`
- `tensor_parallel_size`: `1`（単GPU）
- `gpu_devices`: `all`（または `0`）
- `download_model`: 初回のみ `on`
- `speculative_config`: 最初は `off`（安定確認後に有効化）

---

## UI 項目と実際の対応

### モデル

- 対応: `vllm serve <model_id>`
- 意味: 起動する Hugging Face モデル ID
- 注意:
  - 一覧には「ダウンロード済みモデル」が表示されます。
  - GGUF など形式によっては vLLM 非対応の場合があります。

### コンテキスト長

- 対応: `--max-model-len`
- 意味: 1リクエストで扱う最大トークン長（入力+生成）
- 影響:
  - 大きいほど長文に強い
  - その分 KV キャッシュを消費し、VRAM圧迫・同時実行数低下

### 最大同時リクエスト数

- 対応: `--max-num-seqs`
- 意味: 同時に処理するシーケンス上限
- 影響:
  - 上げるとスループット改善の余地
  - ただし長コンテキストと組み合わせると OOM しやすい

### デフォルト生成パラメータ

以下は `vllm serve` の引数ではなく、`/v1/chat/completions` などで未指定時に backend が補完する既定値です。

- `default_max_tokens`
- `default_temperature`
- `default_top_p`
- `default_frequency_penalty`
- `default_presence_penalty`

### GPU メモリ利用率

- 対応: `--gpu-memory-utilization`
- 意味: KV キャッシュ等に使う VRAM 比率の上限目安
- モード:
  - `auto`: 起動時に空きVRAMから自動計算
  - `manual`: スライダー値を固定で使用
- 目安:
  - 起動失敗する場合は下げる（例: `0.85 -> 0.75 -> 0.70`）

### テンソル並列数

- 対応: `--tensor-parallel-size`
- 意味: モデル重みを何GPUに分割するか
- 注意:
  - `gpu_devices` で見えている GPU 数以下にする

### 使用GPU

- 対応: 起動時 `CUDA_VISIBLE_DEVICES`
- 意味: vLLM プロセスに見せる GPU を制限
- 例:
  - `all`
  - `0`
  - `0,1`

### 起動前にモデルキャッシュを確認/ダウンロード

- 対応: backend 起動フローの `snapshot_download` 実行有無
- 意味:
  - `on`: 未キャッシュなら取得してから起動
  - `off`: 既にキャッシュ済み前提で起動

---

## ツール呼び出し（`enable_auto_tool_choice` / `tool_call_parser`）

クライアントが OpenAI 互換 API で `tool_choice: "auto"` を送る場合、vLLM 側では **自動ツール選択** が有効になっている必要があります。vLLM Manager の「サーバー操作」では次を設定すると、起動コマンドに `--enable-auto-tool-choice` と `--tool-call-parser <名前>` が付きます。

| UI / `config.json` | 実際の vLLM フラグ |
|---------------------|-------------------|
| `enable_auto_tool_choice` をオン | `--enable-auto-tool-choice` |
| `tool_call_parser` に文字列入力 | `--tool-call-parser <値>` |

**`tool_call_parser` に何を書くか**は、「そのモデルが生成するツール呼び出しの**出力形式**」に対応した **vLLM 組み込みパーサの名前**です。モデルファミリーごとに決まっており、**用途が違えば別の文字列**になります。迷ったときの優先順位は次のとおりです。

1. **使っている vLLM のバージョン**で `vllm serve --help` を開き、`--tool-call-parser` の説明を確認する（バージョンで追加・変更があるため）。
2. vLLM 公式の **Tool Calling** の説明で、自分のモデル名が載っているセクションを探す（英語）：[Tool Calling（vLLM）](https://docs.vllm.ai/en/latest/features/tool_calling)。
3. モデル作者（例: Qwen / Nous Hermes）の「Function calling / vLLM」向けドキュメントを参照する。

### 代表的な値（モデル系統の目安）

以下は **よくある対応の例**であり、必ずそのモデルで動く保証ではありません。実際に利用している **モデル ID と vLLM の組み合わせ**に合わせてください。

| 系統・例 | `tool_call_parser` に入れる値の例 |
|---------|-----------------------------------|
| Nous Hermes（Hermes-2-Pro / Hermes-3 など） | `hermes` |
| Qwen2.5 / QwQ など（公式が Hermes 形式と説明しているもの） | `hermes` |
| Meta Llama 3.1 の JSON ツール呼び出し | `llama3_json`（チャットテンプレ調整が必要な場合あり） |
| Llama 4（pythonic 推奨の場合） | `llama4_pythonic` |
| Mistral 公式フォーマット | `mistral` |
| Qwen3-Coder（ドキュメント記載の Coder 向け XML パーサ） | `qwen3_xml` |

**注意:** 以前サンプルとして出ていた `qwen3_coder` という名前は、vLLM のパーサ一覧では **Qwen3-Coder 向けに `qwen3_xml`** と書かれていることが多いです。実際の一覧は **その場の `vllm serve --help`** を正としてください。

### vLLM Manager でまだ渡していないもの

公式ではツール利用時に **`--chat-template`（または `tool_use`）** を追加で指定する例がよく出ます。現状の vLLM Manager は **`--enable-auto-tool-choice` と `--tool-call-parser` のみ**を付与します。Hermes / Qwen などはモデル付属の `tokenizer_config.json` で足りることが多いですが、Llama のように **別テンプレファイルが必須**と書かれているモデルでは、コンテナ内で `vllm serve` を直接叩く・起動ラインを拡張するなどの対応が別途必要になることがあります。

### `config.json` での指定例

`vllm-data` ボリューム内の `config.json` にも同じキーを保存できます。

```json
"enable_auto_tool_choice": true,
"tool_call_parser": "hermes"
```

---

## Speculative Decoding（speculative_config）

`speculative_config` は `vllm serve --speculative-config '<JSON>'` に渡されます。  
目的は「1トークンずつ生成」より効率よくデコードすることです。

### 基本キー

- `method`: 手法
- `num_speculative_tokens`: 1ステップで提案するトークン数
- `rejection_sample_method`: `strict | probabilistic | synthetic`
- `synthetic_acceptance_rate`: synthetic 時のみ

### 手法別の主な使いどころ

- `ngram`
  - 追加モデル不要で試しやすい
  - キー: `prompt_lookup_min`, `prompt_lookup_max`
- `suffix`
  - 追加モデル不要、類似入力が多いと効きやすい
  - キー: `suffix_decoding_max_tree_depth`, `suffix_decoding_max_cached_requests`, `suffix_decoding_max_spec_factor`, `suffix_decoding_min_token_prob`
- `draft_model` / `eagle3`
  - 補助モデルを使う高機能系
  - キー: `model`, `draft_tensor_parallel_size`, `parallel_drafting` など
- `mtp` / `qwen3_next_mtp` / `qwen3_5_mtp`
  - MTP対応モデル向け
  - モデル非対応だと効果が出ない、または起動失敗

### まず試す順番（実運用向け）

1. `speculative_config` を `off` で通常動作を確認  
2. `ngram` で `num_speculative_tokens=2` から開始  
3. 効果が薄ければ `suffix` を試す  
4. MTP対応モデルを使う場合だけ `qwen3_next_mtp` / `qwen3_5_mtp` へ進む

### 代表例

#### ngram の最小例

```json
{
  "method": "ngram",
  "num_speculative_tokens": 2,
  "prompt_lookup_min": 2,
  "prompt_lookup_max": 5
}
```

#### suffix の最小例

```json
{
  "method": "suffix",
  "num_speculative_tokens": 4,
  "suffix_decoding_max_tree_depth": 24,
  "suffix_decoding_max_cached_requests": 10000,
  "suffix_decoding_max_spec_factor": 1.0,
  "suffix_decoding_min_token_prob": 0.1
}
```

#### Qwen MTP 例（対応モデル前提）

```json
{
  "method": "qwen3_next_mtp",
  "num_speculative_tokens": 2
}
```

---

## 失敗したときのチェック

### 起動直後に落ちる

- `context_length` を下げる
- `gpu_memory_utilization` を下げる
- `max_num_seqs` を下げる

### `speculative_config` を有効にすると失敗する

- `method` とモデルの対応を確認
- まず `num_speculative_tokens` を `1` に下げる
- `draft_model` / `eagle3` なら `model` の指定漏れを確認

### 速くならない

- まず通常設定との比較を同一条件で測定
- QPS が高すぎる/短文すぎるケースでは利得が小さいことがある
- 手法を `ngram` ↔ `suffix` で比較し、効く方を使う

---

## 補足

- この画面で設定した値は backend 側の設定として保存され、次回起動時の初期値に使われます。
- 詳細な API 呼び出し例は `docs/api-requests.md` を参照してください。
