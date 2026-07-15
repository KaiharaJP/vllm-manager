# セキュリティ・運用改善（実装済み）

考察で挙がった項目のうち、以下を vLLM Manager に実装済みです。

## 1. WebSocket 認証

- `/ws/metrics` / `/ws/events` は **ログイン token 必須**（クエリ `?token=<Bearer token>`）
- 未認証接続は拒否（code 1008）
- フロントエンド（`useMetricsWebSocket`）は `localStorage` の token を自動付与

**変わること**: 再ビルド後、フロントエンド以外（curl 等）から `/ws/metrics` へ無認証で繋いでも即切断されるようになります。既存のフロントエンド UI は自動対応済みのため、ログイン済みの利用者には見た目上の変化はありません。

## 2. プロンプト内容の漏洩対策

- LiteLLM 追跡イベントの `request_summary` から **ユーザー入力プレビュー（先頭120文字）を除去**
- `litellm_proxy_request` イベントは **管理者ロールのみ** WebSocket で受信
- 管理者は従来どおり REST（リクエスト履歴等）で詳細確認可能

**変わること**: 「モニタリング」タブの一般ユーザー表示から、他利用者のリクエスト内容に関する行が消えます（表示件数が減ります）。管理者向けの「リクエスト履歴」機能自体は変更ありません。

## 3. セッション TTL・ログイン試行制限

| 環境変数 | 既定 | 説明 |
|----------|------|------|
| `VLLM_MANAGER_SESSION_TTL_SEC` | `86400` | 管理 UI セッション有効期限（秒）。0=無期限 |
| `VLLM_MANAGER_LOGIN_MAX_ATTEMPTS` | `5` | ウィンドウ内の最大失敗回数 |
| `VLLM_MANAGER_LOGIN_WINDOW_SEC` | `300` | 失敗回数カウントのウィンドウ |
| `VLLM_MANAGER_LOGIN_LOCKOUT_SEC` | `900` | ロックアウト時間 |

**変わること**: 24時間ログインしっぱなしのブラウザは自動的に再ログインが必要になります。パスワードを5回連続で間違えると15分間ロックされます（正規ユーザーが短時間に何度も打ち間違えると一時的に締め出される点に注意）。

## 4. vLLM 自動復旧

- 起動成功時: インスタンス registry に `auto_restore: true` を保存
- **手動停止**（UI/API）時: `auto_restore: false` に更新
- **backend 再起動**時: `auto_restore=true` かつ未稼働のインスタンスを順次 `start_server` で復旧

**変わること**: `docker compose restart backend` や backend クラッシュ後の自動復帰時に、直前まで稼働していたモデルが自動的に再起動されるようになります（従来は管理者が手動で起動し直す必要がありました）。復旧処理は起動直後に実行されるため、モデルサイズによっては再ロードに数分かかることがあります。

## 5. 監査ログ永続化

- 管理操作イベント（`server_job`, `user_updated`, `litellm_key_updated` 等）を
  `vllm-data/audit.log` に **JSON Lines** で追記
- メトリクス系（`metrics`, `litellm_proxy_request` 等）は監査対象外
- 管理者 API: `GET /api/audit-log?limit=100`

**変わること**: backend を再起動しても「誰が・いつ・何をしたか」の履歴が消えなくなります。`vllm-data` volume の容量が `audit.log` の分だけわずかに増加します。

| 環境変数 | 既定 | 説明 |
|----------|------|------|
| `VLLM_MANAGER_AUDIT_LOG_MAX_BYTES` | `20971520`（20MB） | 1世代あたりの上限サイズ。超えると自動でローテーション |
| `VLLM_MANAGER_AUDIT_LOG_BACKUPS` | `3` | 保持する世代数（`audit.log.1`〜`.N`）。超えた古い世代は削除 |

**変わること（追記）**: 上限サイズを超えると `audit.log` → `audit.log.1` → … と世代がずれ、最も古い世代は自動削除されるため、**ディスクを無制限に消費することがなくなります**。`GET /api/audit-log` は現行ファイルの件数が足りない場合、直近のバックアップ世代からも自動で補完します。

## 6. CORS 設定

| 環境変数 | 説明 |
|----------|------|
| `VLLM_MANAGER_CORS_ORIGINS` | カンマ区切りで許可オリジン指定。未設定または `*` で全開放 |

**変わること**: 既定値は従来どおり全開放のため、`.env` に未設定なら**挙動は変わりません**。設定した場合のみ、指定外オリジンからの API 呼び出しがブラウザ側でブロックされます。

## 7. 弱いデフォルト認証情報の警告

- `app/auth.py` の `collect_security_warnings()` が以下を検出:
  - 管理者アカウントのパスワードが `admin` 等の推測されやすい値と一致
  - `LITELLM_MASTER_KEY` が配布時の既定値 `sk-vllm-default-key` のまま
- 管理者が `/api/auth/login` または `/api/auth/me` を呼ぶと `security_warnings` フィールドで返却
- フロントエンドはダッシュボード上部に警告バナーを表示（管理者のみ）

**変わること**: 初期パスワード（admin/admin）や既定の `LITELLM_MASTER_KEY` を変更しないまま運用していると、**管理者がログインするたびに黄色い警告バナーが表示される**ようになります。一般ユーザー（先生方）には表示されません。パスワードや鍵を変更すれば警告は消えます。

## 8. ヘルスチェック失敗時の通知イベント

- 新規 `app/health_watchdog.py`: 30秒間隔で全稼働インスタンスの health を確認
- **3回連続**でヘルスチェックに失敗したインスタンスについて `instance_unhealthy` イベントを発行
- 復旧時は `instance_health_recovered` イベントを発行
- 自動再起動は行わない（意図しない再起動ループを避けるため、通知のみ）

**変わること**: モデルが応答しなくなった場合、WebSocket イベント（イベント履歴・監査ログ）に `instance_unhealthy` が記録されるようになります。現状は UI 上に専用のアラート表示はまだ無いため、気づくには「ログ」タブやイベント履歴を見る必要があります（自動メール/Slack通知は未実装）。

## 9. Docker コンテナの non-root 実行化

- `Dockerfile.backend`: `vllmapp`（既定 UID/GID 1000）ユーザーを追加
- `docker-entrypoint.sh`: root で起動 → ボリューム権限を調整 → `gosu` で `vllmapp` に降格して実行
- `docker-compose.yml`: `APP_UID` / `APP_GID` ビルド引数を追加（既定 1000、変更可能）

**変わること・注意点（重要）**:

- backend プロセスおよびその子プロセス（vLLM サブプロセス）は非root（UID 1000）で動作するようになります。
- **`vllm-data` ボリュームは起動のたびに再帰的に所有権を揃えます**（比較的小さいため許容範囲）。
- **`hf-cache` ボリュームは容量が大きいため、モデル本体は再帰 chown しません。** ただしダウンロードに必須の **`.locks`（および `hub/.locks`）は起動のたびに `vllmapp` 所有へ揃えます**（旧 root 実行時に残った root 所有ロックが DL 失敗の原因になるため）。
- **`HF_HOME` は `/app/hf-cache` にマウント**します（旧 `/root/.cache/huggingface` から移行。同一 Docker ボリュームのため既存モデルは保持されます）。
- **本 backend は `pid: host` で動作しており、従来は root 権限でホスト上の任意の `vllm serve` プロセス（他のコンテナが起動したものを含む）を停止できていました。non-root 化後は、backend 自身が起動したプロセス（同じ UID）は引き続き停止できますが、root など別ユーザーが所有する外部の vLLM プロセスは停止できなくなります。** `standalone-vllm` profile や他ツールで vLLM を個別起動している場合は影響を確認してください。
- GPU デバイスノード（`/dev/nvidia*`）へのアクセス権限は NVIDIA Container Toolkit の設定に依存します。多くの環境では non-root でも問題なく動作しますが、**未検証のため次回のメンテナンス時間中に必ず動作確認してください**。
- 変更を反映するには `docker compose up -d --build backend` が必要です（イメージの再ビルドが伴うため、既存の起動中モデルは一時停止します）。

このドキュメントの他の変更（1〜8）は API 追加や検知ロジックの追加が中心で後方互換性がありますが、9 のみは **コンテナ実行基盤に関わる破壊的変更になり得るため、本番環境（GPU 稼働中のホスト）へ適用する前に必ずメンテナンス時間を確保し、動作確認してください。**

## 10. GPU VRAM 事前チェック（既存実装の確認）

調査の結果、`app/server_manager.py` の `_preflight_vram_check()` が既に `start_server()` から呼び出されており、`nvidia-smi` で空き VRAM を確認したうえで不足時は起動を中止する仕組みが実装済みでした。考察時点の記載は誤りだったため、ここに追記して訂正します。追加の実装は不要です。

## 11. 初回ログイン時・弱いパスワード利用時の強制変更

- `app/auth.py` の `user_must_change_password()` が、**管理者・一般ユーザーを問わず**、既知の推測されやすいパスワード（`admin` / `password` / `changeme` 等）を使っているアカウントを検出
- `/api/auth/login` と `/api/auth/me` のレスポンスに `must_change_password` フィールドを追加
- フロントエンド（`AuthGate`）は `must_change_password=true` の場合、**他の画面を一切表示せず**パスワード変更フォームをブロッキング表示
- 変更後は既存の `PATCH /api/auth/me` を再利用してパスワードを更新し、即座に通常画面へ遷移

**変わること**: 初期パスワード（`admin`/`admin`）のまま、または `password` 等の推測されやすいパスワードのまま運用しているアカウントは、**ログイン直後に強制的にパスワード変更画面が表示され、変更するまで管理画面や API キー発行画面を使えなくなります**（7. の警告バナーは「表示するだけ」でしたが、本項目は変更を必須にします）。強いパスワードを既に設定しているアカウントには影響ありません。

## 12. 永続 API トークン（PAT）と CLI

ブラウザのセッショントークンは backend 再起動で失効しますが、**永続 API トークン（PAT）** は `vllm-data/api_keys.json` に保存され、再起動後も有効です。

| エンドポイント | メソッド | 認証 | 説明 |
|---|---|---|---|
| `/api/auth/me/tokens` | POST | ログイン済み | PAT 発行（生キーはレスポンスで一度だけ返る） |
| `/api/auth/me/tokens` | GET | ログイン済み | 自分の PAT 一覧（生キーなし） |
| `/api/auth/me/tokens/{token_id}` | DELETE | ログイン済み | PAT 失効 |

- キー形式: `vlmk_` プレフィックス付き。`Authorization: Bearer vlmk_...` で既存の全 admin API（`/api/start`, `/api/stop`, `/api/model-downloads` 等）が利用可能
- 保存されるのは SHA-256 ハッシュのみ（平文は発行時のみ表示）
- 任意で `expires_in_days` を指定可能（未指定=無期限）
- 発行・失効は `audit.log` に `api_token_created` / `api_token_revoked` として記録

**CLI スクリプト**（`scripts/vllm-cli.sh`）:

```bash
# 初回: ログインして PAT を ~/.config/vllm-manager/token に保存
./scripts/vllm-cli.sh token create --name my-automation --username admin --password 'your-password'

# 以降はログイン不要
./scripts/vllm-cli.sh status
./scripts/vllm-cli.sh models list
./scripts/vllm-cli.sh models download Qwen/Qwen2.5-7B-Instruct
./scripts/vllm-cli.sh start Qwen/Qwen2.5-7B-Instruct --context-length 32768
./scripts/vllm-cli.sh stop
./scripts/vllm-cli.sh token list
./scripts/vllm-cli.sh token revoke <token_id>
```

環境変数: `VLLM_MANAGER_URL`（既定 `http://localhost:18000`）、`VLLM_MANAGER_TOKEN`（保存ファイルより優先）。

**変わること**: スクリプトや cron からサーバー起動/停止・モデルダウンロードを自動化できるようになります。PAT は推論用 LiteLLM キー（`/api/auth/me/api-keys`）とは別物です。PAT を漏洩させると admin 相当の操作が可能になるため、ファイル権限（`chmod 600`）と失効運用に注意してください。

### 12-1. PAT のフロントエンドUI・管理者による強制失効

自分の PAT はマイページ（一般ユーザー）/ ユーザー管理画面（管理者）の「永続APIトークン（PAT）」セクションから、curl を使わず GUI で発行・確認・失効できます。発行直後のみ生の値が画面に表示され、以後は再表示できません（prefix のみ表示）。

管理者は「ユーザー管理」で対象ユーザーを選択すると、そのユーザーの PAT 一覧（名前・作成日時・最終使用日時・有効期限・失効状態）を閲覧でき、退職者や漏洩が疑われるトークンを**強制失効**できます（`DELETE /api/users/{username}/tokens/{token_id}`）。管理者は他ユーザーの PAT を代理発行することはできません（発行は本人のみ）。

| エンドポイント | 説明 | 権限 |
|----------------|------|------|
| `GET /api/users/{username}/tokens` | 指定ユーザーの PAT 一覧 | admin only |
| `DELETE /api/users/{username}/tokens/{token_id}` | 指定ユーザーの PAT を強制失効 | admin only |

これにより、これまで「PATを発行した本人にしか停止できない」状態だった問題（アカウント無効化時にPATだけ生き残る等）を解消しています。

## 13. 起動後の自動疎通確認（スモークテスト）

`GET /api/health/check` や `/health` エンドポイントは「プロセスが立っているか」しか確認できず、**実際にモデルが正常に応答を生成できるか**は別問題でした。これを補うため、実際に最小のチャット/embeddingリクエストを送って応答を検証する仕組みを追加しました。

- `app/server_manager.py` の `run_smoke_test(instance_id)`: 稼働中インスタンスへ `/v1/chat/completions`（`task_type=embedding` の場合は `/v1/embeddings`）に最小リクエストを送信し、レイテンシ・生成トークン数・tok/s・応答プレビューを返す
- `POST /api/instances/{instance_id}/smoke-test`（admin only）: 手動実行用エンドポイント。結果は `instance_smoke_test` イベントとして監査ログに記録
- `app/health_watchdog.py`: インスタンスが起動後に**初めて healthy になったタイミングで自動的に1回**疎通テストを実行し、結果をイベント配信（`instance_smoke_test`）。以後は再起動するまで再実行しない
- フロントエンド（`ServerControl.tsx`）の起動中サーバー一覧に「疎通テスト」ボタンを追加。結果（成功時はレイテンシ/tok-s、失敗時はエラー内容）をその場で表示
- CLI: `./scripts/vllm-cli.sh smoke-test <instance_id>`

**変わること**: モデル起動直後に「プロセスは起動しているが実際には応答が返ってこない」状態を自動検知できるようになります。管理者は起動作業のたびに手動でテストプロンプトを送る必要がなくなり、UI上の「疎通テスト」ボタンや自動実行イベントで即座に確認できます。テスト用リクエストは監査ログに記録されるため、モデル切り替え作業の証跡としても使えます。

## 14. GPU/ディスクしきい値監視とアラート

`GET /api/system-metrics` は数値を返すだけで、閾値超過時に気づく仕組みがありませんでした。新規 `app/resource_watchdog.py` が既存の `health_watchdog` と同じ間隔（既定30秒）でホストのGPU/ディスク使用状況を監視し、しきい値を超えたら通知します。

| 監視項目 | 環境変数 | 既定値 |
|----------|----------|--------|
| チェック間隔 | `VLLM_MANAGER_RESOURCE_CHECK_INTERVAL_SEC` | `30`（秒） |
| GPU温度 | `VLLM_MANAGER_GPU_TEMP_ALERT_C` | `85`（℃） |
| GPU VRAM使用率 | `VLLM_MANAGER_GPU_MEMORY_ALERT_PERCENT` | `95`（%） |
| ディスク使用率（root / モデルキャッシュ / 管理データそれぞれ個別） | `VLLM_MANAGER_DISK_ALERT_PERCENT` | `90`（%） |

- しきい値超過時に `resource_alert` イベント、5ポイント以上下回って回復したら `resource_alert_recovered` イベントを発行（監査ログにも記録）
- ヒステリシス（しきい値 −5ポイント）を設けており、しきい値付近の値でアラート発報と回復通知を繰り返す「フラッピング」を防止
- `GET /api/system-metrics` のレスポンスに `disks` フィールドを追加。従来の `disk`（root のみ）に加え、HF キャッシュ（`HF_HOME`）と管理データ（`VLLM_MANAGER_DATA_DIR`）の使用率を個別に確認可能（同一マウントの場合は重複表示しない）
- 「ホーム」タブのシステム監視カードに、上記の各ディスクの使用率が個別表示されるようになりました

**変わること**: GPUが高温になっている、VRAMが逼迫している、モデルキャッシュ用ディスクの空き容量が少なくなっている、といった「サーバーが壊れる/新しいモデルをダウンロードできなくなる前触れ」を、閾値超過時のイベント通知（WebSocket / 監査ログ）で検知できるようになります。自動対処（GPUクロック制限やモデル削除等）は行わず、通知のみです。既定のしきい値で運用に支障がある場合は `.env` の各変数で調整してください。

## 15. バックアップ/リストア

`vllm-data`（ユーザー・PAT・監査ログ・モデルカタログ・インスタンスレジストリ）と LiteLLM の Postgres DB（ユーザー/チーム/virtual key/利用ログ）は、`docker compose down -v` 等でボリュームを消すと**復元不能**でした。バックアップ/リストア用スクリプトを追加しました。

```bash
# バックアップ（./backups/ に保存。既定で7世代分を保持しローテーション）
./scripts/backup-vllm-data.sh

# 特定ディレクトリに保存したい場合
./scripts/backup-vllm-data.sh /path/to/backup-dir

# 復元（先に該当サービスを停止してから実行）
docker compose stop backend
./scripts/restore-vllm-data.sh vllm-data ./backups/vllm-data-20260707-090000.tar.gz
docker compose up -d backend

docker compose stop litellm litellm-gateway
./scripts/restore-vllm-data.sh litellm-db ./backups/litellm-db-20260707-090000.sql.gz
docker compose up -d litellm litellm-gateway
```

- `vllm-data` は `docker run` + `tar` でボリューム全体をアーカイブ（`hf-cache` はモデル重みで巨大かつ再ダウンロード可能なため対象外）
- LiteLLM DB は起動中の `litellm-db` コンテナに対して `pg_dump`/`psql` を実行（コンテナが起動していない場合はスキップ）
- 復元前に確認プロンプトを表示（既存データを完全に上書きするため）
- `BACKUP_DIR` / `BACKUP_RETENTION_COUNT` 環境変数で保存先・保持世代数を調整可能
- バックアップファイルは機密情報（パスワードハッシュ・PATハッシュ・LiteLLM利用ログ等）を含むため、`.gitignore` に `backups/` を追加済み。定期実行する場合は cron 等で `./scripts/backup-vllm-data.sh` を叩き、生成物は社内の安全な場所（暗号化ストレージ等）に転送することを推奨します。

**変わること**: 誤ってボリュームを削除した場合やホスト障害時に、直近のバックアップからユーザーアカウント・APIキー・監査ログ・LiteLLM設定を復元できるようになります。バックアップ自体の定期実行・外部保管は運用側で設定する必要があります（本スクリプトはローカル実行のみで、自動スケジューリングは含みません）。

---

## 16. 管理画面内チャット UI

ログイン後、全ユーザー（admin / 一般）が「チャット」タブからブラウザ上でモデルと会話できます。Chatbox 等の外部アプリをセットアップしなくても、起動中のモデルをその場で試せます。

### 構成

| 項目 | 内容 |
|------|------|
| フロントエンド | `frontend/src/components/ChatPanel.tsx`（ストリーミング表示・会話履歴はブラウザ `localStorage` に保存） |
| バックエンド API | `GET /api/chat/models`、`POST /api/chat/completions`（セッション JWT / PAT で認証） |
| 推論経路 | backend → LiteLLM → backend `/v1/*` プロキシ → vLLM（既存のリクエスト履歴トラッキングに乗る） |
| LiteLLM キー | ユーザーごとに `chat_keys.json` に専用 `sk-` キーを自動発行・保管（ブラウザには渡さない） |

### エンドポイント

```http
GET /api/chat/models
Authorization: Bearer <session JWT または PAT>

POST /api/chat/completions
Authorization: Bearer <session JWT または PAT>
Content-Type: application/json

{"model":"vllm-local","messages":[{"role":"user","content":"こんにちは"}],"stream":true}
```

- `stream: true` の場合は SSE（`text/event-stream`）で応答を逐次返します。
- LiteLLM 側でキーが失効していた場合、バックエンドが自動で再発行して 1 回だけリトライします。

### データの扱い

- **`chat_keys.json`**: `VLLM_MANAGER_DATA_DIR` 配下（`users.json` / `api_keys.json` と同様の機密性）。`scripts/backup-vllm-data.sh` による `vllm-data` ボリュームのバックアップに含まれます。
- **会話履歴**: サーバーには保存されず、各ユーザーのブラウザ `localStorage` のみ。管理者は「リクエスト履歴」タブで LiteLLM 経由の推論ログ（プロンプト含む）を確認できます。
- **初回スコープ**: テキストのみ（画像入力は未対応）。

### 将来の拡張候補

- 画像（Vision）入力
- チャット専用キーへのデフォルト予算 / RPM 制限の管理者設定
- 複数会話の保存・切り替え

---

## 未対応（今後の検討・製品判断が必要な項目）

- TLS 終端（nginx / リバースプロキシ）
- LDAP / SSO 連携
- 不健全インスタンスの自動再起動（現状は通知のみ）や Slack/メール等の外部アラート連携
- コンテンツモデレーション
- マルチノード / 負荷分散、アイドル時の自動アンロード

詳細な背景は計画ドキュメント「vLLM Manager に不足していると考えられる点（考察）」を参照してください。
