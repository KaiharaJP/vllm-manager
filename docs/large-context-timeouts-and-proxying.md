# 大コンテキスト・タイムアウト・プロキシ（サーバー側の整理）

Claude Code や Hermes などから **LiteLLM（例: `:14000`）** 経由で vLLM Manager の **backend（例: `:18000` の `/v1/*`）** を叩き、その先で **vLLM** を動かす構成では、次のような症状が出ることがあります。

| 症状（クライアント側の表現の例） | 典型的な裏側の意味 |
|----------------------------------|-------------------|
| `No response from provider for 300s` | 一定時間、**HTTP レスポンスとして有効なチャンクが届いていない**（ストリームなら最初のイベントが来ない／非ストリームなら本文が揃わない） |
| `APIConnectionError` / `Connection error` | 接続リセット、途中切断、TLS/プロキシ異常、または **サーバーが例外で応答を返し切れなかった** |
| `httpx.ReadTimeout` / `httpcore.ReadTimeout`（backend ログ） | **プロキシが上流からレスポンスを読み切る前に、読み取りタイムアウトで打ち切った** |

このドキュメントでは、**ログで特定しうる原因**と、**サーバー側でどこまで対策できるか**を整理します。

---

## 1. リクエストが流れる経路（ざっくり）

```
クライアント（Claude Code 等）
    → LiteLLM Proxy（:LITELLM_PORT、例 14000）
        → backend の OpenAI 互換プロキシ（Docker 内 http://backend:8000/v1）
            → ホスト上の vLLM（動的ポート、例 8011）
```

- **LiteLLM** は認証・ルーティング・Anthropic 形式の変換などを行うことがあります。
- **backend** は `app/main.py` の `proxy_openai_compat` 等で、**vLLM へ転送**します。
- 追跡用ヘッダ `X-Vllm-Manager-Source: litellm` が付いたリクエストは、`app/litellm_request_track.py` 経由で **メトリクス用に SSE を観測**したり、**非ストリーム時は全文バッファ**したりする経路があります（後述）。

---

## 2. ログから分かった「300秒無応答」の典型原因

### 2.1 backend の `ReadTimeout`（読み取りタイムアウト）

`vllm-manager-backend` のログに、次のようなスタックが出る場合があります。

- `httpx.ReadTimeout` / `httpcore.ReadTimeout`
- 呼び出し元: `proxy_litellm_tracked_v1` → `_buffer_tracked` → `client.request("POST", target, ...)`

**意味:** backend が **上流 vLLM から HTTP レスポンスを読み終える前に**、設定された **read タイムアウト**に達した。

`app/main.py` では、OpenAI 互換プロキシ転送に使う `httpx` のタイムアウトがおおよそ次のように設定されています（変更される可能性があるため、実際の値はコードを確認してください）。

- `httpx.Timeout(300.0, connect=10.0)` のような **全体で数分規模の read** が使われている箇所がある。

**大きいプロンプト（例: 10 万トークン級）**では、vLLM の **prefill（プロンプト処理）だけで数分を超える**ことがあり得ます。その間、クライアントが **ストリーミングで最初のトークンを待っている**場合も、**非ストリームで本文全体を待っている**場合も、「長い無応答」に見えます。backend が先に `ReadTimeout` すると、クライアント側は **接続エラー**や **中途半端な応答**として観測することがあります。

### 2.2 LiteLLM 追跡経路の「全文バッファ」（`stream: false`）

`app/litellm_request_track.py` では、LiteLLM 由来のリクエストに対して:

- `stream: true` → `_stream_tracked`（上流の SSE を観測しつつ転送）
- `stream: false` → `_buffer_tracked`（**上流から返ってきた本文を全部読んでから**クライアントへ返す）

**`stream: false` のとき**、vLLM が **最初のバイトを返すまで**クライアントにも何も返せません。大文脈ではここがボトルネックになりやすく、**クライアントの「N 秒無応答」警告**と相関しやすいです。

### 2.3 vLLM 未起動・接続拒否・503

- backend が `503`（`vLLM server is not running` / `vLLM upstream is unavailable`）を返している場合、クライアントによっては **Connection error** として見えることがあります。
- これは **タイムアウト対策とは別軸**（まず推論サーバーを健全に起動する）の問題です。

### 2.4 前段のリバースプロキシ（nginx 等）

`litellm-gateway`（`:14000`）およびホスト前段の nginx では、

- `proxy_read_timeout`
- `proxy_send_timeout`
- アイドルタイムアウト

が **backend より短い**と、backend 側の設定を伸ばしても **前段で切られる**ことがあります。症状は同様に「長時間無応答のあと切断」になります。

**実装済み（本リポジトリ）:** `config/litellm_gateway.conf` で `proxy_read_timeout` / `proxy_send_timeout` を **600s**。backend は `PROXY_UPSTREAM_READ_TIMEOUT_SEC`（デフォルト 600）で整合。

### 2.5 LiteLLM 経由の `stream` 強制

Hermes 等が `stream: false` を送ると、prefill 完了まで HTTP レスポンスが空のままになり、499/504 の原因になりやすい。

**実装済み:** LiteLLM 由来（`X-Vllm-Manager-Source: litellm`）の `chat/completions` で、`stream` 未指定/false のとき **`stream: true` に上書き**（`PROXY_FORCE_STREAM=true` がデフォルト。`config.json` の `force_stream` でも制御）。無効化は `PROXY_FORCE_STREAM=false`。

---

## 3. サーバー側で「対策できる」こと（優先度の目安）

### 3.1 タイムアウトの整合（最優先で効きやすい）

| レイヤ | 調整の例 |
|--------|----------|
| backend → vLLM の `httpx` read タイムアウト | 巨大 prefill に合わせて **延長**（運用で許容する秒数に） |
| LiteLLM → backend（LiteLLM 側の timeout 設定があれば） | **backend 以上に短くしない**（短い方に合わせて切れる） |
| 前段 nginx / LB | `proxy_read_timeout` 等を **backend 以上**に |

**注意:** 無制限に伸ばすと、**本当にハングしたリクエスト**が長く残り続けるので、**上限＋監視**（ログ、同時接続数）とセットが現実的です。

### 3.2 ストリーミングの徹底（「無応答」対策の本丸）

| 方針 | 効果 |
|------|------|
| 可能な限り **`stream: true`** でクライアント・LiteLLM・backend・vLLMをつなぐ | **最初のチャンクが早く届く**ため、「N 秒まったく何も来ない」状態を避けやすい |
| LiteLLM 追跡で `stream: false` が多い場合の見直し | `_buffer_tracked` による **全文待ち**を減らす |

※ Claude Code / Anthropic 経路では、内部で **Responses API 相当のペイロード**に寄ったり、`stream` の有無がクライアント実装に依存したりします。backend 側で **/v1/responses を chat にブリッジ**するなどの対策を入れている場合は、その経路でも **ストリーム透過**できるかがポイントになります（実装は `app/main.py` を参照）。

### 3.3 推論負荷そのもの（サーバー設定で触れる範囲）

| 項目 | 目的 |
|------|------|
| `max_model_len` / context length | **許容プロンプト長の上限**を現実的に抑え、極端な prefill を防ぐ |
| `max_num_seqs`、同時リクエスト | **キュー待ち**による「何も返らない時間」の延長を抑える |
| モデル・量子化の選択 | 同じ文脈長でも **prefill が短くなる**構成へ寄せる |

### 3.4 運用・可用性

- vLLM の **ヘルスチェック**と、未起動時の **明確な 503**（クライアントがリトライしやすい）
- GPU メモリ不足で起動失敗している場合は、**他プロセスとの VRAM 競合**を解消する（これもサーバー側）

---

## 4. サーバーだけでは「完全には」解決できないこと

| 事象 | 理由 |
|------|------|
| 100k トークン近辺の **純粋な遅さ** | GPU 性能・帯域・モデル・実装の物理上限 |
| クライアント固有の **無応答秒数** | アプリ側のウォッチドッグ（例: 300s）は **クライアント設定**の場合がある |

サーバー側の役割は主に次の二つです。

1. **途中で切らずに**（タイムアウト・プロキシ・バッファリング）推論を完走させられるようにする  
2. **早い段階でストリームを流す**などして、「無応答」と誤判定されないようにする  

---

## 5. 調査するときに見るログ（手早いチェックリスト）

1. **`docker logs vllm-manager-backend`**  
   - `ReadTimeout` / `ConnectTimeout` / `503` / `POST /v1/chat/completions` / `POST /v1/responses`
2. **`docker logs vllm-manager-litellm`**  
   - 上流 URL（`http://backend:8000/v1/...`）、エラー種別、リトライ
3. **前段が nginx の場合**  
   - `error.log` の upstream timed out 等
4. **vLLM 自身のログ**（管理 UI または `vllm.log`）  
   - prefill の長さ、OOM、キュー、起動失敗

---

## 6. 関連ファイル（実装の所在）

| ファイル | 内容の目安 |
|----------|------------|
| `app/main.py` | OpenAI 互換 `/v1/{subpath}` プロキシ、`httpx.Timeout`、ストリーミング透過、Responses ブリッジ等 |
| `app/litellm_request_track.py` | `X-Vllm-Manager-Source: litellm` 付きリクエストの追跡、`_stream_tracked` / `_buffer_tracked` |
| `config/litellm_config.yaml` | LiteLLM の `model_list`、上流 `api_base` 等 |

---

## 7. 関連ドキュメント

- [api-requests.md](./api-requests.md) — ポートと経路の概要  
- [server-start-options.md](./server-start-options.md) — vLLM 起動オプションとトラブルシュート  

---

## 更新履歴

- 初版: 大コンテキスト時のタイムアウト・プロキシ・ストリーミングをサーバー視点で整理
