from __future__ import annotations

import re
from typing import Iterable

_FINAL_NUMBER = re.compile(r"-?\\d+(?:\\.\\d+)?")


def extract_final_number(text: str) -> str | None:
    """Extract the last numeric answer from free-form reasoning text."""
    if not text:
        return None
    if "####" in text:
        text = text.split("####")[-1]
    matches = _FINAL_NUMBER.findall(text.replace(",", ""))
    return matches[-1] if matches else None


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return value.strip()
    return str(int(number)) if number.is_integer() else f"{number:.10g}"


def correctness_reward(completions: Iterable[str], answer: Iterable[str], **_: object) -> list[float]:
    rewards: list[float] = []
    for completion, reference in zip(completions, answer):
        predicted = normalize_number(extract_final_number(str(completion)))
        expected = normalize_number(extract_final_number(str(reference)))
        rewards.append(1.0 if predicted is not None and predicted == expected else 0.0)
    return rewards


def format_reward(completions: Iterable[str], **_: object) -> list[float]:
    """Small shaping reward for clearly marking a final answer."""
    return [0.2 if ("####" in str(c) or "final answer" in str(c).lower()) else 0.0 for c in completions]
