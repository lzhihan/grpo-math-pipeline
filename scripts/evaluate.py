import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_jsonl
from src.metrics import compute_metrics
from src.utils import ensure_dir, load_yaml


def metric_row(name: str, path: str) -> dict:
    rows = load_jsonl(path)
    metrics = compute_metrics(rows)
    return {"model_tag": name, **metrics, "predictions_path": path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--base-preds", default="outputs/baseline/qwen2.5-1.5b-test_predictions.jsonl")
    parser.add_argument("--sft-preds", default="outputs/eval/sft_test_predictions.jsonl")
    parser.add_argument("--grpo-preds", default="outputs/eval/grpo_test_predictions.jsonl")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rows = []

    for name, path in [("base", args.base_preds), ("sft", args.sft_preds), ("grpo", args.grpo_preds)]:
        p = Path(path)
        if p.exists():
            rows.append(metric_row(name, str(p)))
        else:
            print(f"Skip missing file: {p}")

    if not rows:
        raise FileNotFoundError("No prediction files found. Run baseline/SFT/GRPO inference first.")

    df = pd.DataFrame(rows)
    report_csv = Path(cfg["outputs"]["report_csv"])
    ensure_dir(report_csv.parent)
    df.to_csv(report_csv, index=False)
    print(df.to_string(index=False))
    print(f"Saved report: {report_csv}")


if __name__ == "__main__":
    main()
