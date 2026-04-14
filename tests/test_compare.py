from smollest.compare import compare_outputs


def test_exact_match():
    baseline = '{"sentiment": "positive", "confidence": "high"}'
    candidate = '{"sentiment": "positive", "confidence": "high"}'
    result = compare_outputs(baseline, candidate, "test-model")
    assert result.score == 1.0
    assert result.total_fields == 2
    assert len(result.matching_fields) == 2
    assert len(result.mismatched_fields) == 0


def test_partial_match():
    baseline = '{"sentiment": "positive", "confidence": "high"}'
    candidate = '{"sentiment": "positive", "confidence": "low"}'
    result = compare_outputs(baseline, candidate, "test-model")
    assert result.score == 0.5
    assert len(result.matching_fields) == 1
    assert len(result.mismatched_fields) == 1
    assert result.mismatched_fields[0]["field"] == "confidence"


def test_no_match():
    baseline = '{"sentiment": "positive", "topic": "sports"}'
    candidate = '{"sentiment": "negative", "topic": "politics"}'
    result = compare_outputs(baseline, candidate, "test-model")
    assert result.score == 0.0


def test_nested_objects():
    baseline = '{"result": {"label": "spam", "score": 0.9}}'
    candidate = '{"result": {"label": "spam", "score": 0.5}}'
    result = compare_outputs(baseline, candidate, "test-model")
    assert result.score == 0.5
    assert "result.label" in result.matching_fields
    assert result.mismatched_fields[0]["field"] == "result.score"


def test_extra_fields_in_candidate():
    baseline = '{"sentiment": "positive"}'
    candidate = '{"sentiment": "positive", "extra": "field"}'
    result = compare_outputs(baseline, candidate, "test-model")
    assert result.score == 0.5
    assert result.total_fields == 2


def test_missing_fields_in_candidate():
    baseline = '{"sentiment": "positive", "confidence": "high"}'
    candidate = '{"sentiment": "positive"}'
    result = compare_outputs(baseline, candidate, "test-model")
    assert result.score == 0.5


def test_invalid_baseline_json():
    result = compare_outputs("not json", '{"a": 1}', "test-model")
    assert result.score is None
    assert result.error is not None


def test_invalid_candidate_json():
    result = compare_outputs('{"a": 1}', "not json", "test-model")
    assert result.score is None
    assert result.error is not None


def test_empty_objects():
    result = compare_outputs("{}", "{}", "test-model")
    assert result.score == 1.0
    assert result.total_fields == 0


def test_list_values():
    baseline = '{"tags": ["python", "ml"]}'
    candidate = '{"tags": ["python", "ai"]}'
    result = compare_outputs(baseline, candidate, "test-model")
    assert "tags[0]" in result.matching_fields
    assert any(m["field"] == "tags[1]" for m in result.mismatched_fields)
