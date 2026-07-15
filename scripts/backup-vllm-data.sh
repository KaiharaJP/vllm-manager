#!/usr/bin/env bash
# vllm-data / LiteLLM DB backup script.
#
# Backs up:
#   - the `vllm-data` Docker volume (users, API keys, audit log, model catalog,
#     instance registry, config) as a tar.gz
#   - the LiteLLM Postgres database (`litellm-db`) as a pg_dump SQL file, if running
#
# Does NOT back up `hf-cache` (model weights) — it is large and re-downloadable
# from Hugging Face on demand.
#
# Usage:
#   ./scripts/backup-vllm-data.sh [output_dir]
#
# Environment:
#   BACKUP_DIR              Output directory (default: ./backups)
#   BACKUP_RETENTION_COUNT  How many backups to keep per type (default: 7, 0=unlimited)
#   LITELLM_DB_USER / LITELLM_DB_NAME  Used for pg_dump (read from .env if present)

set -euo pipefail

cd "$(dirname "$0")/.."

BACKUP_DIR="${1:-${BACKUP_DIR:-./backups}}"
RETENTION_COUNT="${BACKUP_RETENTION_COUNT:-7}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

LITELLM_DB_USER="${LITELLM_DB_USER:-litellm}"
LITELLM_DB_NAME="${LITELLM_DB_NAME:-litellm}"

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed"
}

need_cmd docker

mkdir -p "$BACKUP_DIR"

find_volume() {
  local name_fragment="$1"
  docker volume ls --format '{{.Name}}' | grep -E "${name_fragment}$" | head -n1
}

rotate_old_backups() {
  local pattern="$1"
  [[ "$RETENTION_COUNT" -gt 0 ]] || return 0
  # shellcheck disable=SC2012
  local files
  files=$(ls -1t "${BACKUP_DIR}"/${pattern} 2>/dev/null || true)
  [[ -n "$files" ]] || return 0
  echo "$files" | tail -n "+$((RETENTION_COUNT + 1))" | while IFS= read -r old; do
    [[ -n "$old" ]] && rm -f "$old" && echo "  古いバックアップを削除しました: $old"
  done
}

echo "=== vllm-data バックアップ ==="
VLLM_DATA_VOLUME="$(find_volume 'vllm-data')"
if [[ -z "$VLLM_DATA_VOLUME" ]]; then
  die "vllm-data volume not found. Run this from the host where 'docker compose up' created it."
fi
echo "volume: ${VLLM_DATA_VOLUME}"

VLLM_DATA_ARCHIVE="${BACKUP_DIR}/vllm-data-${TIMESTAMP}.tar.gz"
docker run --rm \
  -v "${VLLM_DATA_VOLUME}:/data:ro" \
  -v "$(cd "$BACKUP_DIR" && pwd):/backup" \
  alpine:3.20 \
  tar czf "/backup/$(basename "$VLLM_DATA_ARCHIVE")" -C /data .
echo "作成しました: ${VLLM_DATA_ARCHIVE} ($(du -h "$VLLM_DATA_ARCHIVE" | cut -f1))"
rotate_old_backups 'vllm-data-*.tar.gz'

echo
echo "=== LiteLLM DB バックアップ ==="
if docker ps --format '{{.Names}}' | grep -q 'litellm-db'; then
  LITELLM_DB_CONTAINER="$(docker ps --format '{{.Names}}' | grep 'litellm-db' | head -n1)"
  LITELLM_DB_DUMP="${BACKUP_DIR}/litellm-db-${TIMESTAMP}.sql.gz"
  docker exec "$LITELLM_DB_CONTAINER" \
    pg_dump -U "$LITELLM_DB_USER" "$LITELLM_DB_NAME" | gzip >"$LITELLM_DB_DUMP"
  echo "作成しました: ${LITELLM_DB_DUMP} ($(du -h "$LITELLM_DB_DUMP" | cut -f1))"
  rotate_old_backups 'litellm-db-*.sql.gz'
else
  echo "litellm-db コンテナが起動していないためスキップしました。"
fi

echo
echo "バックアップ完了。復元は ./scripts/restore-vllm-data.sh を参照してください。"
