from mvlm.defaults import estimate_cost


def test_known_model_cost():
    cost = estimate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
    assert cost is not None
    assert cost > 0


def test_partial_model_match():
    cost = estimate_cost("gpt-4o-2024-08-06", input_tokens=1000, output_tokens=500)
    assert cost is not None


def test_unknown_model_returns_none():
    cost = estimate_cost("some-random-model", input_tokens=1000, output_tokens=500)
    assert cost is None


def test_zero_tokens():
    cost = estimate_cost("gpt-4o", input_tokens=0, output_tokens=0)
    assert cost == 0.0
