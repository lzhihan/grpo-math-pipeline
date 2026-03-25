import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import save_jsonl
from src.utils import load_yaml


def make_linear_eq(i: int) -> dict:
    # Build solvable equation: ax + b = c
    a = random.randint(1, 9)
    x = random.randint(-20, 20)
    b = random.randint(-20, 20)
    c = a * x + b
    return {
        "id": f"sample-{i:06d}",
        "problem": f"Solve for x: {a}x + {b} = {c}",
        "answer": str(x),
        "metadata": {"source": "synthetic_linear_eq", "difficulty": "easy"},
    }


def build_dataset(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    return [make_linear_eq(i) for i in range(n)]


def split_rows(rows: list[dict], split_cfg: dict, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    return train, val, test


def write_split(rows: list[dict], split_name: str) -> list[dict]:
    out = []
    for row in rows:
        r = dict(row)
        metadata = dict(r.get("metadata", {}))
        metadata["split"] = split_name
        r["metadata"] = metadata
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    num_samples = int(cfg.get("num_samples", 3000))
    debug_samples = int(cfg.get("debug_samples", 32))

    raw_rows = build_dataset(n=num_samples, seed=seed)
    raw_path = Path(cfg["raw_data_path"])
    save_jsonl(raw_path, raw_rows)

    train, val, test = split_rows(raw_rows, cfg["splits"], seed=seed)
    train = write_split(train, "train")
    val = write_split(val, "val")
    test = write_split(test, "test")

    processed_dir = Path(cfg["processed_dir"])
    save_jsonl(processed_dir / "train.jsonl", train)
    save_jsonl(processed_dir / "val.jsonl", val)
    save_jsonl(processed_dir / "test.jsonl", test)

    save_jsonl(cfg["sample_debug_path"], train[:debug_samples])
    print(
        f"Prepared dataset: train={len(train)}, val={len(val)}, test={len(test)}, "
        f"raw={raw_path}"
    )


if __name__ == "__main__":
    main()
