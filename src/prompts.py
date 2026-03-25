SYSTEM_PROMPT = (
    "You are a helpful math tutor. "
    "Solve the problem carefully. "
    "Put the final answer in \\boxed{}."
)


def build_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. "
        "Show your reasoning briefly and put the final answer in \\boxed{}.\n\n"
        f"Problem: {problem}"
    )


def build_messages(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {problem}"},
    ]
