#!/usr/bin/env bash
# vLLM Manager CLI — HTTP API wrapper for server control and model downloads.
#
# Requires: curl, jq
#
# Environment:
#   VLLM_MANAGER_URL      Backend base URL (default: http://localhost:$BACKEND_PORT from .env)
#   VLLM_MANAGER_TOKEN    Bearer token (overrides saved token file)
#   VLLM_MANAGER_USERNAME Login username for `token create` (else .env ADMIN_USER)
#   VLLM_MANAGER_PASSWORD Login password for `token create` (else .env ADMIN_PASSWORD)
#   VLLM_MANAGER_ENV      Explicit path to vllm-manager .env
#   VLLM_MANAGER_CONFIG   Config directory (default: ~/.config/vllm-manager)
#   LITELLM_API_KEY / VLLM_MANAGER_SK_KEY  Inference key override (sk-...)
#   LITELLM_URL           Inference gateway (docs/examples; default :14000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve symlink (skills/.cursor → skills/vllm-manager)
if [[ -L "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
fi

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed"
}

# Load KEY=VALUE from .env without overriding already-exported vars.
# Supports optional single/double quotes; skips comments and blank lines.
load_env_file() {
  local file="$1"
  [[ -f "$file" && -r "$file" ]] || return 1
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
      if [[ "$val" =~ ^\"(.*)\"$ ]]; then
        val="${BASH_REMATCH[1]}"
      elif [[ "$val" =~ ^\'(.*)\'$ ]]; then
        val="${BASH_REMATCH[1]}"
      fi
      if [[ -z "${!key+x}" ]]; then
        export "${key}=${val}"
      fi
    fi
  done <"$file"
  return 0
}

find_dotenv() {
  if [[ -n "${VLLM_MANAGER_ENV:-}" ]]; then
    printf '%s' "$VLLM_MANAGER_ENV"
    return 0
  fi
  # Prefer skills/vllm-manager/.env next to this script package.
  local skill_env
  skill_env="$(cd "${SCRIPT_DIR}/.." && pwd)/.env"
  if [[ -f "$skill_env" ]]; then
    printf '%s' "$skill_env"
    return 0
  fi
  local dir candidate
  dir="$SCRIPT_DIR"
  local i
  for i in 1 2 3 4 5 6; do
    candidate="${dir}/.env"
    if [[ -f "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
    if [[ -f "${dir}/docker-compose.yml" && -f "${dir}/.env.example" && -f "${dir}/.env" ]]; then
      printf '%s' "${dir}/.env"
      return 0
    fi
    [[ "$dir" == "/" ]] && break
    dir="$(dirname "$dir")"
  done
  if [[ -f "${PWD}/.env" ]]; then
    printf '%s' "${PWD}/.env"
    return 0
  fi
  return 1
}

DOTENV_FILE=""
if DOTENV_FILE="$(find_dotenv)"; then
  load_env_file "$DOTENV_FILE" || true
fi

if [[ -z "${VLLM_MANAGER_URL:-}" ]]; then
  if [[ -n "${BACKEND_PORT:-}" ]]; then
    VLLM_MANAGER_URL="http://localhost:${BACKEND_PORT}"
  else
    VLLM_MANAGER_URL="http://localhost:18000"
  fi
fi
if [[ -z "${LITELLM_URL:-}" && -n "${NEXT_PUBLIC_LITELLM_URL:-}" ]]; then
  LITELLM_URL="${NEXT_PUBLIC_LITELLM_URL}"
elif [[ -z "${LITELLM_URL:-}" && -n "${LITELLM_PORT:-}" ]]; then
  LITELLM_URL="http://localhost:${LITELLM_PORT}"
fi

BASE_URL="${VLLM_MANAGER_URL%/}"
CONFIG_DIR="${VLLM_MANAGER_CONFIG:-${HOME}/.config/vllm-manager}"
TOKEN_FILE="${CONFIG_DIR}/token"
SK_KEY_FILE="${CONFIG_DIR}/litellm-key"

load_token() {
  if [[ -n "${VLLM_MANAGER_TOKEN:-}" ]]; then
    printf '%s' "$VLLM_MANAGER_TOKEN"
    return 0
  fi
  if [[ -f "$TOKEN_FILE" ]]; then
    tr -d '\n' <"$TOKEN_FILE"
    return 0
  fi
  return 1
}

save_token() {
  local token="$1"
  mkdir -p "$CONFIG_DIR"
  umask 077
  printf '%s' "$token" >"$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
}

load_sk_key() {
  if [[ -n "${LITELLM_API_KEY:-}" ]]; then
    printf '%s' "$LITELLM_API_KEY"
    return 0
  fi
  if [[ -n "${VLLM_MANAGER_SK_KEY:-}" ]]; then
    printf '%s' "$VLLM_MANAGER_SK_KEY"
    return 0
  fi
  if [[ -f "$SK_KEY_FILE" ]]; then
    tr -d '\n' <"$SK_KEY_FILE"
    return 0
  fi
  return 1
}

save_sk_key() {
  local key="$1"
  mkdir -p "$CONFIG_DIR"
  umask 077
  printf '%s' "$key" >"$SK_KEY_FILE"
  chmod 600 "$SK_KEY_FILE"
}

extract_sk_key() {
  local resp="$1"
  printf '%s' "$resp" | jq -r '.key // .token // .api_key // empty'
}

api_request() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local auth="${4:-yes}"
  local url="${BASE_URL}${path}"
  local -a headers=(-H "Content-Type: application/json")
  local token=""

  if [[ "$auth" == "yes" ]]; then
    token="$(load_token)" || die "No token found. Run: $0 token create --name <name>"
    headers+=(-H "Authorization: Bearer ${token}")
  fi

  local resp http_code tmp
  tmp="$(mktemp)"
  if [[ -n "$body" ]]; then
    http_code="$(curl -sS -o "$tmp" -w '%{http_code}' -X "$method" "$url" "${headers[@]}" -d "$body")"
  else
    http_code="$(curl -sS -o "$tmp" -w '%{http_code}' -X "$method" "$url" "${headers[@]}")"
  fi

  resp="$(cat "$tmp")"
  rm -f "$tmp"

  if [[ "$http_code" -ge 400 ]]; then
    local detail
    detail="$(printf '%s' "$resp" | jq -r '.detail // empty' 2>/dev/null || true)"
    if [[ -z "$detail" ]]; then
      detail="$resp"
    fi
    die "HTTP ${http_code}: ${detail}"
  fi

  printf '%s' "$resp"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  status                              Show primary server status
  servers                             List running vLLM processes
  instances                           List managed instances
  instances stop --id <instance_id>   Stop a managed instance
  start <model_id> [options]          Start vLLM (chat / embedding / rerank)
  stop                                Stop default vLLM server
  restart                             Restart default vLLM server
  smoke-test <instance_id>            Verify instance via chat/embeddings/score
  models list [--task-type <t>]       List catalog (chat|embedding|rerank)
  models register <model_id> [opts]   Register / update catalog entry
  models download <model_id>          Start HF download job
  models downloads                    List download jobs
  models resume <model_id>            Resume stalled download from cache
  models cancel <model_id>            Cancel active downloads for a model
  token create --name <name>          Login and create a persistent API token (vlmk_)
  token list                          List your API tokens
  token revoke <token_id>             Revoke an API token
  inference-key create [options]      Create LiteLLM inference key (sk-); needs PAT
  inference-key ensure [options]      Get-or-create sk- (reuse same models/alias)
  inference-key list                  List your LiteLLM keys
  inference-key delete --key <id>     Delete LiteLLM key (admin PAT required)
  inference-key show                  Print saved/env inference key path status

Start options:
  --context-length <n>                Context length (default: 131072 chat / 8192 pooling)
  --max-num-seqs <n>                  Max concurrent sequences (chat only)
  --task-type <chat|embedding|rerank> Override catalog task_type
  --instance-name <name>              Display name for multi-instance
  --no-download                       Skip model download on start
  --json '<json>'                     Extra ServerStartRequest fields (merged)

models register options:
  --name <display>                    Display name (default: model_id)
  --task-type <chat|embedding|rerank> Required for non-chat (default: chat)
  --size <label>                      Size label (e.g. 7B / embed)
  --context-length <n>                recommended_context_length
  --trust-remote-code                 Enable trust_remote_code
  --gated                             Mark as gated HF model

Token create options:
  --name <name>                       Token display name (required)
  --expires-in-days <n>               Optional expiry in days
  --username <user>                   Override login user
  --password <pass>                   Override login password

  Credential resolution (first match):
    flags → VLLM_MANAGER_USERNAME/PASSWORD → .env VLLM_MANAGER_ADMIN_USER/PASSWORD

inference-key create options:
  --models <m1,m2|*>                  Allowed models (default: * = all; or .env VLLM_MANAGER_INFERENCE_MODELS)
  --save                              Save sk- to ~/.config/vllm-manager/litellm-key

inference-key ensure options:
  --models <m1,m2|*>                  Same as create (default *)
  --force                             Mint a new key even if one already exists
  (always saves to litellm-key; prefers server-side reuse via /api-keys/ensure)

  Model scope (LiteLLM sk- only; PAT has no model list):
    *     all models / aliases (recommended for agents & first setup)
    ids   comma-separated allow-list, e.g. vllm-local,Qwen/Qwen2.5-7B-Instruct

Notes:
  PAT (vlmk_) is for management /api/*. Inference via :14000 needs sk- (separate file).
  Auto-loads skills/vllm-manager/.env first, then walks up, or VLLM_MANAGER_ENV=path.

Environment:
  VLLM_MANAGER_URL                    Backend URL (default from .env BACKEND_PORT)
  VLLM_MANAGER_TOKEN                  Bearer token override (PAT)
  VLLM_MANAGER_CONFIG                 Config dir (default: ~/.config/vllm-manager)
  VLLM_MANAGER_ENV                    Path to .env (optional)
  LITELLM_API_KEY / VLLM_MANAGER_SK_KEY  Inference key override
  LITELLM_URL                         Inference gateway URL for curl examples

Examples:
  $0 token create --name my-automation    # uses .env admin user/password
  $0 inference-key ensure                 # get-or-create models=* (preferred)
  $0 inference-key create --save          # always mint new (prefer ensure)
  $0 inference-key create --models 'vllm-local' --save
  $0 status
  $0 models list --task-type embedding
  $0 models register BAAI/bge-reranker-v2-m3 --task-type rerank --context-length 8192
  $0 models download BAAI/bge-reranker-v2-m3
  $0 start BAAI/bge-reranker-v2-m3 --task-type rerank --context-length 8192 --no-download
  $0 start jinaai/jina-embeddings-v3 --task-type embedding --context-length 8192
  $0 start Qwen/Qwen2.5-7B-Instruct --context-length 32768
  $0 instances
  $0 smoke-test <instance_id>
  $0 stop
EOF
}

cmd_status() {
  api_request GET "/api/status" "" no | jq .
}

cmd_servers() {
  api_request GET "/api/servers" | jq .
}

cmd_instances() {
  local sub="${1:-list}"
  shift || true
  case "$sub" in
    list|"")
      api_request GET "/api/instances" | jq .
      ;;
    stop)
      local instance_id=""
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --id)
            instance_id="$2"
            shift 2
            ;;
          *)
            if [[ -z "$instance_id" && "$1" != --* ]]; then
              instance_id="$1"
              shift
            else
              die "Unknown option: $1"
            fi
            ;;
        esac
      done
      [[ -n "$instance_id" ]] || die "instance_id required (instances stop --id <id>)"
      local body
      body="$(jq -n --arg instance_id "$instance_id" '{instance_id: $instance_id}')"
      api_request POST "/api/instances/stop" "$body" | jq .
      ;;
    *)
      die "Unknown instances subcommand: ${sub}. Use: list | stop"
      ;;
  esac
}

cmd_start() {
  local model_id="${1:-}"
  shift || true
  [[ -n "$model_id" ]] || die "model_id is required"

  local context_length=""
  local max_num_seqs=""
  local task_type=""
  local instance_name=""
  local download_model=true
  local extra_json="{}"
  local context_set=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --context-length)
        context_length="$2"
        context_set=true
        shift 2
        ;;
      --max-num-seqs)
        max_num_seqs="$2"
        shift 2
        ;;
      --task-type)
        task_type="$2"
        case "$task_type" in
          chat|embedding|rerank) ;;
          *) die "--task-type must be chat|embedding|rerank" ;;
        esac
        shift 2
        ;;
      --instance-name)
        instance_name="$2"
        shift 2
        ;;
      --no-download)
        download_model=false
        shift
        ;;
      --json)
        extra_json="$2"
        shift 2
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done

  if [[ "$context_set" != true ]]; then
    if [[ "$task_type" == "embedding" || "$task_type" == "rerank" ]]; then
      context_length=8192
    else
      context_length=131072
    fi
  fi

  local body
  body="$(jq -n \
    --arg model_id "$model_id" \
    --argjson context_length "$context_length" \
    --argjson download_model "$download_model" \
    --arg max_num_seqs "$max_num_seqs" \
    --arg task_type "$task_type" \
    --arg instance_name "$instance_name" \
    --argjson extra "$extra_json" \
    '{
      model_id: $model_id,
      context_length: $context_length,
      download_model: $download_model,
      create_new_instance: true
    }
    + (if ($max_num_seqs | length) > 0 then {max_num_seqs: ($max_num_seqs | tonumber)} else {} end)
    + (if ($task_type | length) > 0 then {task_type: $task_type} else {} end)
    + (if ($instance_name | length) > 0 then {instance_name: $instance_name} else {} end)
    + $extra')"

  api_request POST "/api/start" "$body" | jq .
}

cmd_stop() {
  api_request POST "/api/stop" "{}" | jq .
}

cmd_restart() {
  api_request POST "/api/restart" "{}" | jq .
}

cmd_smoke_test() {
  local instance_id="${1:-}"
  [[ -n "$instance_id" ]] || die "instance_id is required (see: $0 instances)"
  api_request POST "/api/instances/${instance_id}/smoke-test" "{}" | jq .
}

cmd_models_list() {
  local task_type=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --task-type)
        task_type="$2"
        case "$task_type" in
          chat|embedding|rerank) ;;
          *) die "--task-type must be chat|embedding|rerank" ;;
        esac
        shift 2
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
  local resp
  resp="$(api_request GET "/api/models" "" no)"
  if [[ -n "$task_type" ]]; then
    printf '%s' "$resp" | jq --arg t "$task_type" '[.[] | select((.task_type // "chat") == $t)]'
  else
    printf '%s' "$resp" | jq .
  fi
}

cmd_models_register() {
  local model_id="${1:-}"
  shift || true
  [[ -n "$model_id" ]] || die "model_id is required"

  local name=""
  local task_type="chat"
  local size="unknown"
  local context_length=""
  local trust_remote_code=false
  local gated=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name)
        name="$2"
        shift 2
        ;;
      --task-type)
        task_type="$2"
        case "$task_type" in
          chat|embedding|rerank) ;;
          *) die "--task-type must be chat|embedding|rerank" ;;
        esac
        shift 2
        ;;
      --size)
        size="$2"
        shift 2
        ;;
      --context-length)
        context_length="$2"
        shift 2
        ;;
      --trust-remote-code)
        trust_remote_code=true
        shift
        ;;
      --gated)
        gated=true
        shift
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done

  if [[ -z "$context_length" ]]; then
    if [[ "$task_type" == "embedding" || "$task_type" == "rerank" ]]; then
      context_length=8192
    else
      context_length=8192
    fi
  fi
  if [[ -z "$name" ]]; then
    name="$model_id"
  fi

  local body
  body="$(jq -n \
    --arg id "$model_id" \
    --arg name "$name" \
    --arg size "$size" \
    --arg task_type "$task_type" \
    --argjson recommended_context_length "$context_length" \
    --argjson trust_remote_code "$trust_remote_code" \
    --argjson gated "$gated" \
    '{
      id: $id,
      name: $name,
      size: $size,
      task_type: $task_type,
      recommended_context_length: $recommended_context_length,
      trust_remote_code: $trust_remote_code,
      gated: $gated
    }')"
  api_request POST "/api/models" "$body" | jq .
}

cmd_models_download() {
  local model_id="${1:-}"
  [[ -n "$model_id" ]] || die "model_id is required"
  local body
  body="$(jq -n --arg model_id "$model_id" '{model_id: $model_id, force: false}')"
  api_request POST "/api/model-downloads" "$body" | jq .
}

cmd_models_downloads() {
  api_request GET "/api/model-downloads" | jq .
}

cmd_models_resume() {
  local model_id="${1:-}"
  [[ -n "$model_id" ]] || die "model_id is required"
  local body
  body="$(jq -n --arg model_id "$model_id" '{model_id: $model_id}')"
  api_request POST "/api/model-downloads/resume" "$body" | jq .
}

cmd_models_cancel() {
  local model_id="${1:-}"
  [[ -n "$model_id" ]] || die "model_id is required"
  local body
  body="$(jq -n --arg model_id "$model_id" '{model_id: $model_id}')"
  api_request POST "/api/model-downloads/cancel" "$body" | jq .
}

cmd_token_create() {
  local name=""
  local expires_in_days=""
  local username="${VLLM_MANAGER_USERNAME:-${VLLM_MANAGER_ADMIN_USER:-}}"
  local password="${VLLM_MANAGER_PASSWORD:-${VLLM_MANAGER_ADMIN_PASSWORD:-}}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name)
        name="$2"
        shift 2
        ;;
      --expires-in-days)
        expires_in_days="$2"
        shift 2
        ;;
      --username)
        username="$2"
        shift 2
        ;;
      --password)
        password="$2"
        shift 2
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done

  [[ -n "$name" ]] || die "--name is required"
  if [[ -z "$username" || -z "$password" ]]; then
    local hint="set VLLM_MANAGER_ADMIN_USER/PASSWORD in .env"
    if [[ -n "${DOTENV_FILE:-}" ]]; then
      hint="loaded ${DOTENV_FILE} but admin credentials missing; or pass --username/--password"
    else
      hint="no .env found (set VLLM_MANAGER_ENV=/path/to/vllm-manager/.env) or pass --username/--password"
    fi
    die "username/password required (${hint})"
  fi

  local login_body session_token create_body resp raw_token
  login_body="$(jq -n --arg username "$username" --arg password "$password" '{username: $username, password: $password}')"
  resp="$(api_request POST "/api/auth/login" "$login_body" no)"
  session_token="$(printf '%s' "$resp" | jq -r '.token')"

  if [[ -n "$expires_in_days" ]]; then
    create_body="$(jq -n --arg name "$name" --argjson expires_in_days "$expires_in_days" '{name: $name, expires_in_days: $expires_in_days}')"
  else
    create_body="$(jq -n --arg name "$name" '{name: $name}')"
  fi

  local tmp http_code
  tmp="$(mktemp)"
  http_code="$(curl -sS -o "$tmp" -w '%{http_code}' -X POST "${BASE_URL}/api/auth/me/tokens" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${session_token}" \
    -d "$create_body")"
  resp="$(cat "$tmp")"
  rm -f "$tmp"

  if [[ "$http_code" -ge 400 ]]; then
    local detail
    detail="$(printf '%s' "$resp" | jq -r '.detail // empty' 2>/dev/null || true)"
    [[ -n "$detail" ]] || detail="$resp"
    die "HTTP ${http_code}: ${detail}"
  fi

  raw_token="$(printf '%s' "$resp" | jq -r '.token // empty')"
  [[ -n "$raw_token" ]] || die "Failed to create token: no token in response"

  save_token "$raw_token"
  echo "Persistent API token created and saved to ${TOKEN_FILE}"
  printf '%s\n' "$resp" | jq 'del(.token) + {token_saved: true, token_prefix: .prefix}'
}

cmd_token_list() {
  api_request GET "/api/auth/me/tokens" | jq .
}

cmd_token_revoke() {
  local token_id="${1:-}"
  [[ -n "$token_id" ]] || die "token_id is required"
  api_request DELETE "/api/auth/me/tokens/${token_id}" | jq .
}

cmd_inference_key_create() {
  # Default: all models (*). Override via --models or VLLM_MANAGER_INFERENCE_MODELS in .env.
  local models="${VLLM_MANAGER_INFERENCE_MODELS:-*}"
  local do_save=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --models)
        models="${2:-}"
        shift 2
        ;;
      --save)
        do_save=1
        shift
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
  [[ -n "$models" ]] || die "--models must not be empty"

  local create_body resp raw_key
  create_body="$(jq -n --arg models "$models" '
    ($models | split(",") | map(gsub("^\\s+|\\s+$"; "")) | map(select(length > 0))) as $m
    | {models: (if ($m | length) == 0 then ["*"] else $m end)}
  ')"

  resp="$(api_request POST "/api/auth/me/api-keys" "$create_body")"
  raw_key="$(extract_sk_key "$resp")"
  [[ -n "$raw_key" ]] || die "Failed to create inference key: no key in response"

  if [[ "$do_save" -eq 1 ]]; then
    save_sk_key "$raw_key"
    echo "LiteLLM inference key created (models=${models}) and saved to ${SK_KEY_FILE}"
    printf '%s\n' "$resp" | jq --arg path "$SK_KEY_FILE" --arg prefix "${raw_key:0:8}" --arg models "$models" '
      del(.key, .token, .api_key) + {key_saved: true, key_file: $path, key_prefix: $prefix, models: ($models | split(",") )}
    '
  else
    printf '%s\n' "$resp" | jq .
    echo "Tip: re-run with --save to store in ${SK_KEY_FILE} (raw sk- is only shown once)." >&2
  fi
}

cmd_inference_key_ensure() {
  local models="${VLLM_MANAGER_INFERENCE_MODELS:-*}"
  local force=0
  local alias="${VLLM_MANAGER_INFERENCE_KEY_ALIAS:-vllm-cli}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --models)
        models="${2:-}"
        shift 2
        ;;
      --alias)
        alias="${2:-}"
        shift 2
        ;;
      --force)
        force=1
        shift
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
  [[ -n "$models" ]] || die "--models must not be empty"
  [[ -n "$alias" ]] || die "--alias must not be empty"

  local force_json=false
  [[ "$force" -eq 1 ]] && force_json=true

  local body resp raw_key reused
  body="$(jq -n --arg models "$models" --arg alias "$alias" --argjson force "$force_json" '
    ($models | split(",") | map(gsub("^\\s+|\\s+$"; "")) | map(select(length > 0))) as $m
    | {
        models: (if ($m | length) == 0 then ["*"] else $m end),
        key_alias: $alias,
        force: $force
      }
  ')"

  resp="$(api_request POST "/api/auth/me/api-keys/ensure" "$body")"
  raw_key="$(extract_sk_key "$resp")"
  [[ -n "$raw_key" ]] || die "Failed to ensure inference key: no key in response"
  reused="$(printf '%s' "$resp" | jq -r '.reused // false')"

  save_sk_key "$raw_key"
  if [[ "$reused" == "true" ]]; then
    echo "Reused inference key (alias=${alias}, models=${models}) → ${SK_KEY_FILE}"
  else
    echo "Created inference key (alias=${alias}, models=${models}) → ${SK_KEY_FILE}"
  fi
  printf '%s\n' "$resp" | jq --arg path "$SK_KEY_FILE" --arg prefix "${raw_key:0:8}" '
    del(.key, .token, .api_key) + {key_saved: true, key_file: $path, key_prefix: $prefix}
  '
}

cmd_inference_key_list() {
  api_request GET "/api/auth/me/api-keys" | jq .
}

cmd_inference_key_delete() {
  local key_id=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --key)
        key_id="${2:-}"
        shift 2
        ;;
      *)
        if [[ -z "$key_id" && "$1" != --* ]]; then
          key_id="$1"
          shift
        else
          die "Unknown option: $1"
        fi
        ;;
    esac
  done
  [[ -n "$key_id" ]] || die "--key <hash_or_sk> is required (admin PAT required)"

  local body
  body="$(jq -n --arg k "$key_id" '{payload: {keys: [$k]}}')"
  api_request POST "/api/litellm/keys/delete" "$body" | jq .
}

cmd_inference_key_show() {
  local key=""
  if key="$(load_sk_key)"; then
    local source="file"
    if [[ -n "${LITELLM_API_KEY:-}" ]]; then
      source="LITELLM_API_KEY"
    elif [[ -n "${VLLM_MANAGER_SK_KEY:-}" ]]; then
      source="VLLM_MANAGER_SK_KEY"
    else
      source="$SK_KEY_FILE"
    fi
    jq -n --arg source "$source" --arg prefix "${key:0:8}" \
      '{found: true, source: $source, key_prefix: $prefix}'
  else
    jq -n --arg path "$SK_KEY_FILE" \
      '{found: false, hint: ("run inference-key create --save or set LITELLM_API_KEY"), expected_file: $path}'
  fi
}

main() {
  need_cmd curl
  need_cmd jq

  local cmd="${1:-}"
  shift || true

  case "$cmd" in
    status)
      cmd_status
      ;;
    servers)
      cmd_servers
      ;;
    instances)
      cmd_instances "$@"
      ;;
    start)
      cmd_start "$@"
      ;;
    stop)
      cmd_stop
      ;;
    restart)
      cmd_restart
      ;;
    smoke-test)
      cmd_smoke_test "$@"
      ;;
    models)
      local sub="${1:-}"
      shift || true
      case "$sub" in
        list)
          cmd_models_list "$@"
          ;;
        register)
          cmd_models_register "$@"
          ;;
        download)
          cmd_models_download "$@"
          ;;
        downloads)
          cmd_models_downloads
          ;;
        resume)
          cmd_models_resume "$@"
          ;;
        cancel)
          cmd_models_cancel "$@"
          ;;
        *)
          die "Unknown models subcommand: ${sub:-<none>}. Use: list | register | download | downloads | resume | cancel"
          ;;
      esac
      ;;
    token)
      local sub="${1:-}"
      shift || true
      case "$sub" in
        create)
          cmd_token_create "$@"
          ;;
        list)
          cmd_token_list
          ;;
        revoke)
          cmd_token_revoke "$@"
          ;;
        *)
          die "Unknown token subcommand: ${sub:-<none>}. Use: create | list | revoke"
          ;;
      esac
      ;;
    inference-key)
      local sub="${1:-}"
      shift || true
      case "$sub" in
        create)
          cmd_inference_key_create "$@"
          ;;
        ensure)
          cmd_inference_key_ensure "$@"
          ;;
        list)
          cmd_inference_key_list
          ;;
        delete)
          cmd_inference_key_delete "$@"
          ;;
        show)
          cmd_inference_key_show
          ;;
        *)
          die "Unknown inference-key subcommand: ${sub:-<none>}. Use: create | ensure | list | delete | show"
          ;;
      esac
      ;;
    help|-h|--help|"")
      usage
      ;;
    *)
      die "Unknown command: $cmd. Run '$0 help' for usage."
      ;;
  esac
}

main "$@"
