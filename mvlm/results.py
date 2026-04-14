from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from mvlm.compare import ComparisonResult


def log_result(
    baseline_model: str,
    baseline_content: str,
    comparison: ComparisonResult,
    log_file: str,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_model": baseline_model,
        "candidate": comparison.candidate,
        "score": comparison.score,
        "total_fields": comparison.total_fields,
        "matching_fields": comparison.matching_fields,
        "mismatched_fields": comparison.mismatched_fields,
        "error": comparison.error,
    }

    existing = []
    if os.path.exists(log_file):
        try:
            with open(log_file) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(entry)
    with open(log_file, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def print_comparison(
    baseline_model: str,
    comparisons: list[ComparisonResult],
) -> None:
    print(f"\n{'=' * 60}")
    print(f"mvlm comparison — baseline: {baseline_model}")
    print(f"{'=' * 60}")

    for comp in comparisons:
        if comp.error:
            print(f"  {comp.candidate}: ERROR — {comp.error}")
        else:
            score_pct = f"{comp.score * 100:.0f}%" if comp.score is not None else "N/A"
            match_str = f"{len(comp.matching_fields)}/{comp.total_fields} fields"
            print(f"  {comp.candidate}: {score_pct} match ({match_str})")
            if comp.mismatched_fields:
                for m in comp.mismatched_fields:
                    print(
                        f"    ✗ {m['field']}: "
                        f"baseline={m['baseline']!r} vs candidate={m['candidate']!r}"
                    )

    print(f"{'=' * 60}\n")


def report(log_file: str = "mvlm_results.json") -> None:
    if not os.path.exists(log_file):
        print("No results found. Run some comparisons first.")
        return

    with open(log_file) as f:
        entries = json.load(f)

    candidates: dict[str, list[float]] = {}
    for entry in entries:
        name = entry["candidate"]
        score = entry.get("score")
        if score is not None:
            candidates.setdefault(name, []).append(score)

    print(f"\n{'=' * 60}")
    print(f"mvlm summary — {len(entries)} comparisons")
    print(f"{'=' * 60}")

    for name, scores in sorted(candidates.items()):
        avg = sum(scores) / len(scores)
        print(f"  {name}: {avg * 100:.1f}% avg match ({len(scores)} calls)")

    print(f"{'=' * 60}\n")
