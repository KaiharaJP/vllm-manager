#!/usr/bin/env bash
# Restore vllm-data volume and/or LiteLLM DB from a backup created by
# scripts/backup-vllm-data.sh.
#
# IMPORTANT: Stop the backend (and litellm, for DB restore) before running
# this, otherwise the running process may overwrite files during restore or
# hold a DB connection open.
#
# Usage:
#   ./scripts/restore-vllm-data.sh vllm-data <path/to/vllm-data-*.tar.gz>
#   ./scripts/restore-vllm-data.sh litellm-db <path/to/litellm-db-*.sql.gz>

set -euo pipefail

cd "$(dirname "$0")/.."

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed"
}

need_cmd docker

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

LITELLM_DB_USER="${LITELLM_DB_USER:-litellm}"
LITELLM_DB_NAME="${LITELLM_DB_NAME:-litellm}"

find_volume() {
  local name_fragment="$1"
  docker volume ls --format '{{.Name}}' | grep -E "${name_fragment}$" | head -n1
}

confirm() {
  local prompt="$1"
  read -r -p "${prompt} [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

restore_vllm_data() {
  local archive="$1"
  [[ -f "$archive" ]] || die "backup file not found: $archive"

  local volume
  volume="$(find_volume 'vllm-data')"
  [[ -n "$volume" ]] || die "vllm-data volume not found"

  echo "復元先ボリューム: ${volume}"
  echo "警告: 現在の users.json / api_keys.json / audit.log / instances.json 等がすべて上書きされます。"
  echo "backend を事前に停止してください（docker compose stop backend）。"
  confirm "本当に復元しますか？" || { echo "中止しました。"; exit 1; }

  docker run --rm \
    -v "${volume}:/data" \
    -v "$(cd "$(dirname "$archive")" && pwd):/backup:ro" \
    alpine:3.20 \
    sh -c "rm -rf /data/* /data/.[!.]* 2>/dev/null; tar xzf /backup/$(basename "$archive") -C /data"

  echo "復元完了しました。backend を再起動してください（docker compose up -d backend）。"
}

restore_litellm_db() {
  local dump="$1"
  [[ -f "$dump" ]] || die "backup file not found: $dump"

  if ! docker ps --format '{{.Names}}' | grep -q 'litellm-db'; then
    die "litellm-db container is not running. Start it first: docker compose up -d litellm-db"
  fi
  local container
  container="$(docker ps --format '{{.Names}}' | grep 'litellm-db' | head -n1)"

  echo "復元先: ${container} (db=${LITELLM_DB_NAME}, user=${LITELLM_DB_USER})"
  echo "警告: 現在の LiteLLM ユーザー/チーム/APIキー/利用ログがすべて上書きされます。"
  echo "litellm サービスを事前に停止してください（docker compose stop litellm litellm-gateway）。"
  confirm "本当に復元しますか？" || { echo "中止しました。"; exit 1; }

  gunzip -c "$dump" | docker exec -i "$container" psql -U "$LITELLM_DB_USER" -d "$LITELLM_DB_NAME"
  echo "復元完了しました。litellm を再起動してください（docker compose up -d litellm litellm-gateway）。"
}

usage() {
  cat <<EOF
Usage:
  $(basename "$0") vllm-data <path/to/vllm-data-*.tar.gz>
  $(basename "$0") litellm-db <path/to/litellm-db-*.sql.gz>
EOF
}

case "${1:-}" in
  vllm-data)
    [[ -n "${2:-}" ]] || { usage; exit 1; }
    restore_vllm_data "$2"
    ;;
  litellm-db)
    [[ -n "${2:-}" ]] || { usage; exit 1; }
    restore_litellm_db "$2"
    ;;
  *)
    usage
    exit 1
    ;;
esac
