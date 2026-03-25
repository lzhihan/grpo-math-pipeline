import json
from pathlib import Path
from typing import Iterable

from src.prompts import build_prompt


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_sft_records(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        prompt = build_prompt(row["problem"])
        completion = f"Final answer: \\boxed{{{row['answer']}}}"
        records.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "completion": completion,
                "answer": row["answer"],
            }
        )
    return records
