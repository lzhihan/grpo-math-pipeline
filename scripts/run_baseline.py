import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answer_extractor import extract_answer, normalize_answer
from src.dataset import load_jsonl, save_jsonl
from src.metrics import compute_metrics
from src.prompts import build_prompt
from src.utils import ensure_dir, load_yaml


def resolve_model_name(model_arg: str) -> str:
    p = Path(model_arg)
    # Absolute/relative filesystem path should exist if used as local model.
    if p.is_absolute() or model_arg.startswith("."):
        if not p.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {p}\n"
                "If you want to load from Hugging Face Hub, use a repo id like "
                "'Qwen/Qwen2.5-1.5B-Instruct' instead of a filesystem path."
            )
        return str(p)
    return model_arg


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_file = cfg[f"{args.split}_file"]
    rows = load_jsonl(data_file)

    gen_cfg = cfg["generation"]
    model_name = resolve_model_name(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    default_out = cfg["outputs"]["baseline_predictions"]
    out_path = Path(args.output) if args.output else Path(default_out)
    ensure_dir(out_path.parent)

    predictions = []
    for row in tqdm(rows, desc=f"Baseline-{args.split}"):
        prompt = build_prompt(row["problem"])
        model_output = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            do_sample=bool(gen_cfg["do_sample"]),
            temperature=float(gen_cfg["temperature"]),
        )
        pred = extract_answer(model_output)
        gold = normalize_answer(row["answer"])
        predictions.append(
            {
                "id": row["id"],
                "problem": row["problem"],
                "answer": gold,
                "prompt": prompt,
                "model_output": model_output,
                "predicted_answer": pred,
                "is_correct": pred == gold,
                "model_name": model_name,
            }
        )

    save_jsonl(out_path, predictions)
    metrics = compute_metrics(predictions)
    print(f"Saved predictions: {out_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
