# vLLM Manager — Cursor Skills package

Other projects can reuse vLLM Manager operations (start/stop **chat / embedding / rerank**, downloads, inference) by installing this skill for Cursor agents.

## Contents

```
skills/vllm-manager/
├── SKILL.md           # Agent instructions (read first)
├── reference.md       # API, auth, ports, task_type routing
├── examples.md        # Copy-paste workflows (LLM / embed / rerank)
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

```bash
export VLLM_MANAGER_URL=http://<your-host>:18000

~/.cursor/skills/vllm-manager/scripts/vllm-cli.sh token create \
  --name my-project --username admin --password '<password>'
```

### Capability map

| Need | How |
|------|-----|
| Start chat LLM | `start <id> --context-length …` |
| Start embedding | `start <id> --task-type embedding --context-length 8192` |
| Start reranker (vLLM-native) | `start <id> --task-type rerank --context-length 8192` |
| Register catalog entry | `models register <id> --task-type …` |
| Download / resume | `models download` / `models resume` |
| Smoke test | `smoke-test <instance_id>` |
| Inference chat | `:14000/v1/chat/completions` + `sk-`（LiteLLM） |
| Inference embedding / rerank | `:14000/v1/embeddings` / `/v1/rerank` / `/v1/score`（gateway→Manager。固定モデル登録不要） |
| CrossEncoder runtime | **Not supported** (Phase 2) |

See `SKILL.md` and `examples.md` for full workflows.

## This repo

`.cursor/skills/vllm-manager` symlinks to `skills/vllm-manager` so agents in this workspace load the skill automatically.
