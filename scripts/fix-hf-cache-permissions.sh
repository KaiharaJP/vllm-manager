#!/usr/bin/env bash
# HF キャッシュの書き込み権限を修復する（backend コンテナ内で root 実行）。
# 用途: ダウンロードが Permission denied (.locks) で失敗したときの緊急対応。
set -euo pipefail

HF_DIR="${HF_HOME:-/app/hf-cache}"
MODEL_ID="${1:-}"

echo "Fixing HF cache permissions under ${HF_DIR} ..."

mkdir -p "${HF_DIR}/.locks" "${HF_DIR}/hub/.locks"
chown vllmapp:vllmapp "${HF_DIR}"
chmod u+rwx "${HF_DIR}"
chown -R vllmapp:vllmapp "${HF_DIR}/.locks"
if [ -d "${HF_DIR}/hub/.locks" ]; then
  chown -R vllmapp:vllmapp "${HF_DIR}/hub/.locks"
fi

if [ -n "${MODEL_ID}" ]; then
  model_dir="${HF_DIR}/models--${MODEL_ID//\//--}"
  hub_model_dir="${HF_DIR}/hub/models--${MODEL_ID//\//--}"
  for path in "${model_dir}" "${hub_model_dir}"; do
    if [ -d "${path}" ]; then
      echo "Fixing partial cache: ${path}"
      chown -R vllmapp:vllmapp "${path}"
    fi
  done
fi

echo "Done."
