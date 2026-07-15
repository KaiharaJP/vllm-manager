# vLLM Manager — Cursor Skills package

Other projects can reuse vLLM Manager operations (start/stop **chat / embedding / rerank**, downloads, inference) by installing this skill for Cursor agents.

## Contents

```
skills/vllm-manager/
├── README.md          # This file (human install / configure)
├── SKILL.md           # Agent instructions (read first)
├── reference.md       # API, auth, ports, task_type routing
├── examples.md        # Copy-paste workflows (LLM / embed / rerank)
├── .env.example       # Template (URLs, admin login, inference models=*)
├── .env               # Local copy (gitignored) — edit after install
└── scripts/
    └── vllm-cli.sh    # CLI wrapper (curl + jq)
```

`scripts/vllm-cli.sh` is a copy of the repo's `scripts/vllm-cli.sh`. When updating the main CLI, sync with:

```bash
cp scripts/vllm-cli.sh skills/vllm-manager/scripts/vllm-cli.sh
```

## Install

### Personal (all projects)

```bash
cp -r skills/vllm-manager ~/.cursor/skills/vllm-manager
```

### Single project

```bash
mkdir -p .cursor/skills
cp -r /path/to/vllm-manager/skills/vllm-manager .cursor/skills/
```

## Configure

Edit `.env` in this directory (copy from `.env.example` if needed). The CLI loads this file first.

Defaults on this host are typically:

| Variable | Example |
|----------|---------|
| `VLLM_MANAGER_URL` | `http://hinton.kanazawa-it.ac.jp:18000` |
| `LITELLM_URL` | `http://hinton.kanazawa-it.ac.jp:14000` |
| `VLLM_MANAGER_INFERENCE_MODELS` | `*` (all models) |

```bash
CLI=./scripts/vllm-cli.sh   # from this directory
# or: skills/vllm-manager/scripts/vllm-cli.sh from repo root

$CLI token create --name my-project
$CLI inference-key ensure          # models=*; reuses if already issued
```

- `token create` → management PAT (`vlmk_`) — **no** model list (uses `.env` admin user/password)
- `inference-key ensure` → LiteLLM sk- — default **all models** (`*`); same models/alias → reuse (no duplicates)

## Capability map

| Need | How |
|------|-----|
| Start chat LLM | `start <id> --context-length …` |
| Start embedding | `start <id> --task-type embedding --context-length 8192` |
| Start reranker (vLLM-native) | `start <id> --task-type rerank --context-length 8192` |
| Register catalog entry | `models register <id> --task-type …` |
| Download / resume | `models download` / `models resume` |
| Smoke test | `smoke-test <instance_id>` |
| Inference chat | `:14000/v1/chat/completions` + `sk-`（LiteLLM） |
| Inference embedding / rerank | `:14000/v1/embeddings` / `/v1/rerank` / `/v1/score` |
| CrossEncoder runtime | **Not supported** (Phase 2) |

See `SKILL.md` and `examples.md` for full workflows.

## This repo

`.cursor/skills/vllm-manager` symlinks to `skills/vllm-manager` so agents in this workspace load the skill automatically.
