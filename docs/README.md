# ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [teacher-guide.md](./teacher-guide.md) | 先生方（一般ユーザー）向け利用ガイド。ログイン、API キーの受け取り方、チャットアプリ（Chatbox）の設定、API リクエストの具体例（curl / Python） |
| [security-and-operations.md](./security-and-operations.md) | セキュリティ強化・監査ログ・vLLM 自動復旧など運用上の改善点と設定 |
| [api-requests.md](./api-requests.md) | Docker 経由で公開される API へのリクエスト方法（curl / 認証 / エンドポイント） |
| [server-start-options.md](./server-start-options.md) | `サーバー管理` の起動オプション詳細（各項目の意味、推奨値、`tool_call_parser` / ツール呼び出し、speculative_config、トラブルシュート） |
| [large-context-timeouts-and-proxying.md](./large-context-timeouts-and-proxying.md) | 大コンテキスト時の「300秒無応答」・`ReadTimeout`・接続エラー: 経路・ログの見方・サーバー側でできる対策の整理 |
| [mtp-investigation-report.md](./mtp-investigation-report.md) | MTP（投機的デコード）の tok/s・受理率の調査、50 tok/s からの加速見積もり |
| [../skills/README.md](../skills/README.md) | Cursor Agent Skill パッケージ（他プロジェクトへのインストール方法） |
