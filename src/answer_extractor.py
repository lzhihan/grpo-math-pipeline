import re

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
NUMERIC_PATTERN = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")


def normalize_answer(answer: str) -> str:
    value = answer.strip().replace(" ", "")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def extract_boxed_answer(text: str) -> str | None:
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return normalize_answer(matches[-1])


def extract_answer(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    fallback = NUMERIC_PATTERN.findall(text)
    if not fallback:
        return None
    return normalize_answer(fallback[-1])
