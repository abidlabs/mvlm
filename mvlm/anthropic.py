from __future__ import annotations

import os

from mvlm.candidates import run_candidates
from mvlm.compare import ComparisonResult, compare_outputs
from mvlm.results import log_result, print_comparison


def _anthropic_to_openai_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        converted.append({"role": msg["role"], "content": content})
    return converted


class _Messages:
    def __init__(self, wrapper: Anthropic):
        self._wrapper = wrapper

    def create(self, **kwargs):
        response = self._wrapper._client.messages.create(**kwargs)

        if not self._wrapper._candidates:
            return response

        messages = kwargs.get("messages", [])
        system = kwargs.get("system")

        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(_anthropic_to_openai_messages(messages))

        baseline_content = ""
        for block in response.content:
            if block.type == "text":
                baseline_content += block.text

        candidate_results = run_candidates(
            messages=openai_messages,
            candidates=self._wrapper._candidates,
            hf_token=self._wrapper._hf_token,
        )

        comparisons = []
        for cr in candidate_results:
            if cr.error:
                comp = ComparisonResult(candidate=cr.candidate, error=cr.error)
            else:
                comp = compare_outputs(
                    baseline=baseline_content,
                    candidate_content=cr.content,
                    candidate_name=cr.candidate,
                )
            comparisons.append(comp)

        baseline_model = kwargs.get("model", "unknown")
        print_comparison(baseline_model, comparisons)

        if self._wrapper._log_file:
            for comp in comparisons:
                log_result(
                    baseline_model, baseline_content, comp, self._wrapper._log_file
                )

        return response


class Anthropic:
    def __init__(
        self,
        candidates: list[str] | None = None,
        hf_token: str | None = None,
        log_file: str | None = "mvlm_results.json",
        **kwargs,
    ):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install it with: pip install mvlm[anthropic]"
            )

        self._client = _anthropic.Anthropic(**kwargs)
        self._candidates = candidates or []
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._log_file = log_file
        self.messages = _Messages(self)

    def __getattr__(self, name):
        return getattr(self._client, name)
