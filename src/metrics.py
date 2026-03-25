from statistics import mean


def is_correct(predicted: str | None, gold: str) -> bool:
    return predicted is not None and predicted == gold


def compute_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "boxed_rate": 0.0,
            "avg_output_tokens_rough": 0.0,
        }

    correct = []
    boxed = []
    lengths = []
    for row in rows:
        predicted = row.get("predicted_answer")
        gold = row["answer"]
        output_text = row.get("model_output", "")
        correct.append(is_correct(predicted, gold))
        boxed.append("\\boxed{" in output_text)
        lengths.append(len(output_text.split()))

    return {
        "num_samples": len(rows),
        "accuracy": sum(correct) / len(correct),
        "boxed_rate": sum(boxed) / len(boxed),
        "avg_output_tokens_rough": mean(lengths),
    }
