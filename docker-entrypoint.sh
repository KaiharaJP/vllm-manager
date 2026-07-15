#!/bin/bash
# backend コンテナのエントリポイント。
# root で起動し、ボリュームの書き込み権限を非rootユーザー（vllmapp）に揃えた後、
# gosu で権限を落としてからアプリを実行する。
set -e

DATA_DIR="${VLLM_MANAGER_DATA_DIR:-/app/data}"
HF_DIR="${HF_HOME:-/app/hf-cache}"

mkdir -p "$DATA_DIR" "$HF_DIR" "$HF_DIR/.locks" "$HF_DIR/hub" "$HF_DIR/hub/.locks" "$HF_DIR/xet" "$HF_DIR/xet/logs"

# vllm-data は比較的小さいため再帰的に所有権を揃える。
chown -R vllmapp:vllmapp "$DATA_DIR" 2>/dev/null || true

# hf-cache 本体は大きいため再帰 chown はしないが、ダウンロードに必須の
# ディレクトリだけは毎回 vllmapp 所有に揃える（旧 root 実行時に root 所有が残ると DL 失敗する）。
chown vllmapp:vllmapp "$HF_DIR" 2>/dev/null || true
chmod u+rwx "$HF_DIR" 2>/dev/null || true
chown -R vllmapp:vllmapp "$HF_DIR/.locks" 2>/dev/null || true
if [ -d "$HF_DIR/hub" ]; then
  chown vllmapp:vllmapp "$HF_DIR/hub" 2>/dev/null || true
  chmod u+rwx "$HF_DIR/hub" 2>/dev/null || true
fi
if [ -d "$HF_DIR/hub/.locks" ]; then
  chown -R vllmapp:vllmapp "$HF_DIR/hub/.locks" 2>/dev/null || true
fi
if [ -d "$HF_DIR/xet" ]; then
  chown vllmapp:vllmapp "$HF_DIR/xet" 2>/dev/null || true
  chmod u+rwx "$HF_DIR/xet" 2>/dev/null || true
fi
if [ -d "$HF_DIR/xet/logs" ]; then
  chown -R vllmapp:vllmapp "$HF_DIR/xet/logs" 2>/dev/null || true
fi

_ensure_traversable() {
    local dir
    dir="$(cd "$1" 2>/dev/null && pwd)" || return 0
    while [ "$dir" != "/" ] && [ -n "$dir" ]; do
        chmod o+x "$dir" 2>/dev/null || true
        dir="$(dirname "$dir")"
    done
}
_ensure_traversable "$DATA_DIR"

exec gosu vllmapp "$@"
