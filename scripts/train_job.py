#!/usr/bin/env python3
"""
学習ジョブランナー（LoRA SFT / DPO / GRPO）。

training_manager からサブプロセスとして起動される。app パッケージには依存しない。
進捗は job-dir/status.json に書き込み、backend の API がそれを読んで返す。

データセット形式（JSONL、1 行 1 サンプル）:
  sft:  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", ...}]}
        または {"text": "..."}
  dpo:  {"prompt": "...", "chosen": "...", "rejected": "..."}
  grpo: {"prompt": "..."} + reward.type に応じて "answer"（exact_match）/ "keywords"（contains）
"""

import argparse
import json
import time
from pathlib import Path


def write_status(job_dir: Path, **fields):
    path = job_dir / "status.json"
    try:
        current = json.loads(path.read_text())
    except Exception:
        current = {}
    current.update(fields)
    current["updated_at"] = time.time()
    path.write_text(json.dumps(current, ensure_ascii=False))


def load_jsonl_dataset(path_or_id: str):
    from datasets import Dataset, load_dataset

    if path_or_id.endswith(".jsonl"):
        rows = []
        with open(path_or_id) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return Dataset.from_list(rows)
    return load_dataset(path_or_id, split="train")


def build_model_kwargs(quantization: str):
    import torch

    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    return kwargs


def build_lora_config(hp: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=int(hp.get("lora_r", 16)),
        lora_alpha=int(hp.get("lora_alpha", 32)),
        lora_dropout=float(hp.get("lora_dropout", 0.05)),
        target_modules=hp.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


def make_progress_callback(job_dir: Path):
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            total = state.max_steps or 0
            write_status(
                job_dir,
                status="running",
                step=state.global_step,
                total_steps=total,
                epoch=round(state.epoch or 0.0, 3),
                loss=logs.get("loss"),
                reward=logs.get("reward"),
                progress=round(state.global_step / total, 4) if total else None,
            )

    return ProgressCallback()


def completion_to_text(completion) -> str:
    # GRPO の completion は文字列 or chat メッセージ列
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for m in completion:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
        return "\n".join(parts)
    return str(completion)


def build_reward_funcs(reward: dict):
    rtype = str(reward.get("type") or "")

    if rtype == "exact_match":
        # データセットの "answer" 列と完全一致（前後空白は無視）で 1.0
        def exact_match(completions, answer=None, **kwargs):
            answers = answer or [""] * len(completions)
            return [
                1.0 if completion_to_text(c).strip() == str(a).strip() else 0.0
                for c, a in zip(completions, answers)
            ]

        return [exact_match]

    if rtype == "contains":
        # データセットの "keywords" 列（list[str]）の含有率を報酬にする
        def contains(completions, keywords=None, **kwargs):
            kw_lists = keywords or [[] for _ in completions]
            scores = []
            for c, kws in zip(completions, kw_lists):
                text = completion_to_text(c)
                kws = kws or []
                scores.append(sum(1.0 for k in kws if str(k) in text) / max(1, len(kws)))
            return scores

        return [contains]

    if rtype == "remote":
        # 外部報酬サーバーへ POST {prompts, completions} → {"rewards": [...]}
        import httpx

        url = str(reward["url"])
        timeout = float(reward.get("timeout_sec", 120))

        def remote(prompts, completions, **kwargs):
            payload = {
                "prompts": [completion_to_text(p) for p in prompts],
                "completions": [completion_to_text(c) for c in completions],
            }
            resp = httpx.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            rewards = resp.json()["rewards"]
            return [float(r) for r in rewards]

        return [remote]

    raise ValueError(f"unsupported reward type: {rtype!r}")


def run(job_dir: Path):
    config = json.loads((job_dir / "config.json").read_text())
    method = config["method"]
    hp = config.get("hyperparams") or {}
    output_dir = config["output_dir"]

    write_status(job_dir, status="running", started_at=time.time())

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = load_jsonl_dataset(config["dataset"])
    model_kwargs = build_model_kwargs(config.get("quantization", "4bit"))
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    peft_config = build_lora_config(hp)
    callback = make_progress_callback(job_dir)

    common_args = dict(
        output_dir=str(job_dir / "checkpoints"),
        num_train_epochs=float(hp.get("epochs", 1)),
        per_device_train_batch_size=int(hp.get("batch_size", 1)),
        gradient_accumulation_steps=int(hp.get("grad_accum", 8)),
        learning_rate=float(hp.get("learning_rate", 2e-4 if method == "sft" else 5e-6)),
        logging_steps=int(hp.get("logging_steps", 5)),
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=bool(hp.get("gradient_checkpointing", True)),
        report_to=[],
    )
    max_len = int(hp.get("max_seq_len", 2048))

    if method == "sft":
        from trl import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=peft_config,
            args=SFTConfig(max_length=max_len, **common_args),
            callbacks=[callback],
        )
    elif method == "dpo":
        from trl import DPOConfig, DPOTrainer

        trainer = DPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=peft_config,
            args=DPOConfig(
                max_length=max_len,
                beta=float(hp.get("dpo_beta", 0.1)),
                **common_args,
            ),
            callbacks=[callback],
        )
    elif method == "grpo":
        from trl import GRPOConfig, GRPOTrainer

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=peft_config,
            reward_funcs=build_reward_funcs(config.get("reward") or {}),
            args=GRPOConfig(
                max_completion_length=int(hp.get("max_completion_length", 512)),
                num_generations=int(hp.get("num_generations", 4)),
                **common_args,
            ),
            callbacks=[callback],
        )
    else:
        raise ValueError(f"unsupported method: {method!r}")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_status(
        job_dir,
        status="completed",
        finished_at=time.time(),
        adapter_path=output_dir,
        progress=1.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    job_dir = Path(args.job_dir)
    try:
        run(job_dir)
    except Exception as e:
        import traceback

        traceback.print_exc()
        write_status(job_dir, status="failed", error=f"{type(e).__name__}: {e}", finished_at=time.time())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
