# mvlm

Find the minimum viable language model for your task. Drop-in replacements for OpenAI/Anthropic clients that silently replay requests to smaller LLMs and compare structured outputs.

## Install

```bash
pip install mvlm[openai]       # for OpenAI
pip install mvlm[anthropic]    # for Anthropic
pip install mvlm[all]          # both
```

## Usage

```python
from mvlm import openai

client = openai.OpenAI(
    candidates=[
        "mistralai/Mistral-7B-Instruct-v0.3",  # HuggingFace model
        "http://localhost:1234/v1",               # local server (LM Studio, llama.cpp, etc.)
    ],
    hf_token="hf_...",  # or set HF_TOKEN env var
)

result = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Classify as positive/negative: I love this!"}],
)
# Returns the normal OpenAI response
# Prints comparison scores for each candidate to console
# Logs detailed results to mvlm_results.json
```

Works the same way with Anthropic:

```python
from mvlm import anthropic

client = anthropic.Anthropic(
    candidates=["mistralai/Mistral-7B-Instruct-v0.3"],
)

result = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Classify as positive/negative: I love this!"}],
)
```

## How it works

1. Your API call goes to the baseline model (GPT-4o, Claude, etc.) as normal
2. The same messages are replayed to each candidate model (HuggingFace serverless or local OpenAI-compatible server)
3. Structured outputs (JSON) are compared field-by-field with exact match
4. Results are printed and logged so you can find which smaller model matches your baseline

Remote candidates run in parallel; local candidates run sequentially to avoid memory issues.

## View results

```python
import mvlm
mvlm.report()  # prints summary across all logged comparisons
```

## License

MIT
