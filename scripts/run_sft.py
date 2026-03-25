import argparse
import sys
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_jsonl
from src.prompts import build_prompt
from src.utils import load_yaml


def resolve_model_name(model_arg: str, cfg_model: str) -> str:
    model_name = model_arg.strip() if model_arg else cfg_model
    p = Path(model_name)
    if p.is_absolute() or model_name.startswith("."):
        if not p.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {p}\n"
                "If you want to load from Hugging Face Hub, use a repo id like "
                "'Qwen/Qwen2.5-1.5B-Instruct' instead of a filesystem path."
            )
    return model_name


def to_text_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        prompt = build_prompt(row["problem"])
        completion = f"Final answer: \\boxed{{{row['answer']}}}"
        out.append({"text": f"{prompt}\n\n{completion}"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_name = resolve_model_name(args.model, cfg["model_name_or_path"])

    train_rows = load_jsonl(cfg["train_file"])
    val_rows = load_jsonl(cfg["val_file"])
    train_ds = Dataset.from_list(to_text_rows(train_rows))
    val_ds = Dataset.from_list(to_text_rows(val_rows))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        eval_strategy="steps",
        eval_steps=int(cfg["eval_steps"]),
        bf16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"SFT model saved to: {cfg['output_dir']}")


if __name__ == "__main__":
    main()
