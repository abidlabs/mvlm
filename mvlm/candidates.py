from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class CandidateResult:
    candidate: str
    content: str | None = None
    error: str | None = None
    latency_ms: float = 0.0


def _is_local_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def _is_url(candidate: str) -> bool:
    return candidate.startswith("http://") or candidate.startswith("https://")


def _run_openai_compat(
    candidate: str, messages: list[dict], model: str | None = None
) -> CandidateResult:
    try:
        import openai as _openai
    except ImportError:
        return CandidateResult(
            candidate=candidate,
            error="openai package not installed (needed for OpenAI-compatible servers)",
        )

    start = time.monotonic()
    try:
        client = _openai.OpenAI(base_url=candidate, api_key="not-needed")
        response = client.chat.completions.create(
            model=model or "default",
            messages=messages,
        )
        content = response.choices[0].message.content
        elapsed = (time.monotonic() - start) * 1000
        return CandidateResult(candidate=candidate, content=content, latency_ms=elapsed)
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return CandidateResult(candidate=candidate, error=str(e), latency_ms=elapsed)


def _run_hf_inference(
    candidate: str, messages: list[dict], hf_token: str | None = None
) -> CandidateResult:
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return CandidateResult(
            candidate=candidate,
            error="huggingface-hub package not installed",
        )

    start = time.monotonic()
    try:
        client = InferenceClient(model=candidate, token=hf_token)
        response = client.chat_completion(messages=messages)
        content = response.choices[0].message.content
        elapsed = (time.monotonic() - start) * 1000
        return CandidateResult(candidate=candidate, content=content, latency_ms=elapsed)
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        return CandidateResult(candidate=candidate, error=str(e), latency_ms=elapsed)


def run_candidates(
    messages: list[dict],
    candidates: list[str],
    hf_token: str | None = None,
) -> list[CandidateResult]:
    local_candidates = []
    remote_candidates = []

    for c in candidates:
        if _is_url(c) and _is_local_url(c):
            local_candidates.append(c)
        else:
            remote_candidates.append(c)

    results: list[CandidateResult] = []

    if remote_candidates:
        with ThreadPoolExecutor(max_workers=len(remote_candidates)) as executor:
            futures = {}
            for c in remote_candidates:
                if _is_url(c):
                    fut = executor.submit(_run_openai_compat, c, messages)
                else:
                    fut = executor.submit(_run_hf_inference, c, messages, hf_token)
                futures[fut] = c

            for fut in as_completed(futures):
                results.append(fut.result())

    for c in local_candidates:
        results.append(_run_openai_compat(c, messages))

    return results
