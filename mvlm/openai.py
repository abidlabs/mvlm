from __future__ import annotations

import os

from mvlm.candidates import run_candidates
from mvlm.compare import compare_outputs
from mvlm.results import log_result, print_comparison


class _Completions:
    def __init__(self, wrapper: OpenAI):
        self._wrapper = wrapper

    def create(self, **kwargs):
        response = self._wrapper._client.chat.completions.create(**kwargs)

        if not self._wrapper._candidates:
            return response

        messages = kwargs.get("messages", [])
        baseline_content = response.choices[0].message.content

        candidate_results = run_candidates(
            messages=messages,
            candidates=self._wrapper._candidates,
            hf_token=self._wrapper._hf_token,
        )

        comparisons = []
        for cr in candidate_results:
            if cr.error:
                from mvlm.compare import ComparisonResult

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


class _Chat:
    def __init__(self, wrapper: OpenAI):
        self.completions = _Completions(wrapper)


class OpenAI:
    def __init__(
        self,
        candidates: list[str] | None = None,
        hf_token: str | None = None,
        log_file: str | None = "mvlm_results.json",
        **kwargs,
    ):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "openai package is required. Install it with: pip install mvlm[openai]"
            )

        self._client = _openai.OpenAI(**kwargs)
        self._candidates = candidates or []
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._log_file = log_file
        self.chat = _Chat(self)

    def __getattr__(self, name):
        return getattr(self._client, name)
