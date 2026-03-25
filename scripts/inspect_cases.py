import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_jsonl


def to_map(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def improved_cases(before_rows: list[dict], after_rows: list[dict]) -> list[dict]:
    before = to_map(before_rows)
    after = to_map(after_rows)
    out = []
    for rid, a in after.items():
        b = before.get(rid)
        if b and (not b.get("is_correct", False)) and a.get("is_correct", False):
            out.append({"id": rid, "problem": a["problem"], "before": b["model_output"], "after": a["model_output"]})
    return out


def regressed_cases(before_rows: list[dict], after_rows: list[dict]) -> list[dict]:
    before = to_map(before_rows)
    after = to_map(after_rows)
    out = []
    for rid, a in after.items():
        b = before.get(rid)
        if b and b.get("is_correct", False) and (not a.get("is_correct", False)):
            out.append({"id": rid, "problem": a["problem"], "before": b["model_output"], "after": a["model_output"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="e.g. baseline predictions jsonl")
    parser.add_argument("--after", required=True, help="e.g. grpo predictions jsonl")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    before_rows = load_jsonl(args.before)
    after_rows = load_jsonl(args.after)

    improved = improved_cases(before_rows, after_rows)
    regressed = regressed_cases(before_rows, after_rows)

    print(f"Improved: {len(improved)}")
    for row in improved[: args.limit]:
        print(f"- {row['id']}: {row['problem']}")

    print(f"Regressed: {len(regressed)}")
    for row in regressed[: args.limit]:
        print(f"- {row['id']}: {row['problem']}")


if __name__ == "__main__":
    main()
