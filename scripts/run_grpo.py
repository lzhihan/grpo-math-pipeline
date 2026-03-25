import argparse
import inspect
import json
import sys
from pathlib import Path

from datasets import Dataset
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_jsonl
from src.prompts import build_prompt
from src.rewards import correctness_reward, format_reward
from src.utils import load_yaml


def resolve_model_and_tokenizer_source(model_name_or_path: str):
    model_path = Path(model_name_or_path)
    adapter_cfg_path = model_path / "adapter_config.json"
    if model_path.exists() and adapter_cfg_path.exists():
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        base_model = adapter_cfg.get("base_model_name_or_path", "")
        tokenizer_source = model_name_or_path
        if not (model_path / "tokenizer_config.json").exists() and base_model:
            tokenizer_source = base_model

        print(f"[INFO] Detected PEFT adapter model for GRPO init: {model_name_or_path}")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            is_trainable=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        return peft_model, tokenizer_source

    return model_name_or_path, model_name_or_path


def to_grpo_rows(rows: list[dict]) -> list[dict]:
    return [{"prompt": build_prompt(r["problem"]), "answer": r["answer"], "id": r["id"]} for r in rows]


def correctness_reward_adapter(completions, answer, **kwargs):
    if completions and isinstance(completions[0], list):
        texts = [c[0]["content"] for c in completions]
    else:
        texts = completions
    answers = answer if isinstance(answer, list) else [answer] * len(texts)
    return correctness_reward(texts, answers)


def format_reward_adapter(completions, **kwargs):
    if completions and isinstance(completions[0], list):
        texts = [c[0]["content"] for c in completions]
    else:
        texts = completions
    return format_reward(texts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    train_rows = load_jsonl(cfg["train_file"])
    val_rows = load_jsonl(cfg["val_file"])
    train_ds = Dataset.from_list(to_grpo_rows(train_rows))
    val_ds = Dataset.from_list(to_grpo_rows(val_rows))

    model_for_trainer, tokenizer_source = resolve_model_and_tokenizer_source(cfg["model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    desired_grpo_kwargs = {
        "output_dir": cfg["output_dir"],
        "learning_rate": float(cfg["learning_rate"]),
        "num_train_epochs": float(cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "num_generations": int(cfg["num_generations"]),
        "max_prompt_length": int(cfg["max_prompt_length"]),
        "max_completion_length": int(cfg["max_completion_length"]),
        "logging_steps": int(cfg["logging_steps"]),
        "save_steps": int(cfg["save_steps"]),
        "save_total_limit": int(cfg.get("save_total_limit", 2)),
        "eval_steps": int(cfg["eval_steps"]),
        "bf16": True,
        "report_to": [],
    }
    grpo_config_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys()) - {"self"}
    if "eval_strategy" in grpo_config_params:
        desired_grpo_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in grpo_config_params:
        desired_grpo_kwargs["evaluation_strategy"] = "steps"
    grpo_kwargs = {k: v for k, v in desired_grpo_kwargs.items() if k in grpo_config_params}
    dropped_config_keys = [k for k in desired_grpo_kwargs if k not in grpo_kwargs]
    if dropped_config_keys:
        print(f"[INFO] GRPOConfig ignored unsupported keys for this trl version: {dropped_config_keys}")
    grpo_args = GRPOConfig(**grpo_kwargs)

    trainer_kwargs = {
        "model": model_for_trainer,
        "reward_funcs": [correctness_reward_adapter, format_reward_adapter],
        "args": grpo_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "processing_class": tokenizer,
    }
    trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters.keys()) - {"self"}
    for name in ("num_generations", "max_prompt_length", "max_completion_length"):
        if name in trainer_params and name in cfg:
            trainer_kwargs[name] = int(cfg[name])
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"GRPO model saved to: {cfg['output_dir']}")


if __name__ == "__main__":
    main()
