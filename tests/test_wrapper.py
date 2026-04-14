from unittest.mock import MagicMock, patch


def _make_openai_response(content: str, prompt_tokens=10, completion_tokens=5):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


@patch("openai.OpenAI")
def test_openai_wrapper_returns_original(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    expected = _make_openai_response('{"sentiment": "positive"}')
    mock_client.chat.completions.create.return_value = expected

    from mvlm.openai import OpenAI

    client = OpenAI(candidates=[], api_key="test")
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test"}],
    )
    assert result is expected


@patch("openai.OpenAI")
def test_openai_wrapper_calls_candidates(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    baseline_response = _make_openai_response('{"sentiment": "positive"}')
    mock_client.chat.completions.create.return_value = baseline_response

    from mvlm.openai import OpenAI

    with patch("mvlm.openai.run_candidates") as mock_run:
        from mvlm.candidates import CandidateResult

        mock_run.return_value = [
            CandidateResult(
                candidate="test-model",
                content='{"sentiment": "positive"}',
                latency_ms=100,
            )
        ]

        client = OpenAI(
            candidates=["test-model"],
            project="test-project",
            api_key="test",
        )
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "classify this"}],
        )

        assert result is baseline_response
        mock_run.assert_called_once()


@patch("openai.OpenAI")
def test_openai_wrapper_uses_default_candidates(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    from mvlm.defaults import DEFAULT_CANDIDATES
    from mvlm.openai import OpenAI

    client = OpenAI(api_key="test")
    assert client._candidates == DEFAULT_CANDIDATES


@patch("openai.OpenAI")
def test_openai_wrapper_per_call_candidates(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    baseline_response = _make_openai_response('{"a": "b"}')
    mock_client.chat.completions.create.return_value = baseline_response

    from mvlm.openai import OpenAI

    with patch("mvlm.openai.run_candidates") as mock_run:
        from mvlm.candidates import CandidateResult

        mock_run.return_value = [
            CandidateResult(
                candidate="override-model", content='{"a": "b"}', latency_ms=50
            )
        ]

        client = OpenAI(candidates=["default-model"], api_key="test")
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            candidates=["override-model"],
        )

        call_args = mock_run.call_args
        assert call_args.kwargs["candidates"] == ["override-model"]


@patch("openai.OpenAI")
def test_openai_wrapper_proxies_attributes(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.models = MagicMock()
    mock_openai_cls.return_value = mock_client

    from mvlm.openai import OpenAI

    client = OpenAI(candidates=[], api_key="test")
    _ = client.models
    assert _ is mock_client.models
