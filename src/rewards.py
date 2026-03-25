from src.answer_extractor import extract_answer, normalize_answer


def correctness_reward(completions: list[str], answers: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answers):
        pred = extract_answer(completion)
        rewards.append(1.0 if pred == normalize_answer(gold) else 0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    return [0.05 if "\\boxed{" in completion else 0.0 for completion in completions]
