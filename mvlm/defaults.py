from __future__ import annotations

DEFAULT_CANDIDATES = [
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
]

COST_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-20250514": {"input": 0.80, "output": 4.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = COST_PER_1M_TOKENS.get(model)
    if pricing is None:
        for key, val in COST_PER_1M_TOKENS.items():
            if key in model or model in key:
                pricing = val
                break
    if pricing is None:
        return None
    return (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000
