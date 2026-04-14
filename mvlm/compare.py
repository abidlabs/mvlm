from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ComparisonResult:
    candidate: str
    score: float | None = None
    total_fields: int = 0
    matching_fields: list[str] = field(default_factory=list)
    mismatched_fields: list[dict] = field(default_factory=list)
    error: str | None = None


def _flatten_fields(
    obj: dict | list | str | int | float | bool | None,
    prefix: str = "",
) -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                flat.update(_flatten_fields(value, full_key))
            else:
                flat[full_key] = value
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            full_key = f"{prefix}[{i}]"
            if isinstance(value, (dict, list)):
                flat.update(_flatten_fields(value, full_key))
            else:
                flat[full_key] = value
    else:
        flat[prefix] = obj
    return flat


def compare_outputs(
    baseline: str, candidate_content: str, candidate_name: str
) -> ComparisonResult:
    try:
        baseline_json = json.loads(baseline)
    except (json.JSONDecodeError, TypeError):
        return ComparisonResult(
            candidate=candidate_name,
            error="Failed to parse baseline output as JSON",
        )

    try:
        candidate_json = json.loads(candidate_content)
    except (json.JSONDecodeError, TypeError):
        return ComparisonResult(
            candidate=candidate_name,
            error="Failed to parse candidate output as JSON",
        )

    baseline_flat = _flatten_fields(baseline_json)
    candidate_flat = _flatten_fields(candidate_json)

    all_keys = set(baseline_flat.keys()) | set(candidate_flat.keys())
    if not all_keys:
        return ComparisonResult(candidate=candidate_name, score=1.0, total_fields=0)

    matching = []
    mismatched = []

    for key in sorted(all_keys):
        baseline_val = baseline_flat.get(key)
        candidate_val = candidate_flat.get(key)
        if baseline_val == candidate_val:
            matching.append(key)
        else:
            mismatched.append(
                {
                    "field": key,
                    "baseline": baseline_val,
                    "candidate": candidate_val,
                }
            )

    return ComparisonResult(
        candidate=candidate_name,
        score=len(matching) / len(all_keys),
        total_fields=len(all_keys),
        matching_fields=matching,
        mismatched_fields=mismatched,
    )
