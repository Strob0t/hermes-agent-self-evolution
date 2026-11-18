"""Tests for tolerant test-case parsing and retrying generation.

No network: the dspy generator is replaced by a stub that returns canned text.
"""

import json

import pytest

from evolution.core import dataset_builder
from evolution.core.dataset_builder import (
    MAX_GENERATION_ATTEMPTS,
    MIN_VALID_CASES,
    SyntheticDatasetBuilder,
    parse_test_cases,
    _remove_trailing_commas,
)
from evolution.core.config import EvolutionConfig


def _case(i: int) -> dict:
    return {
        "task_input": f"input {i}",
        "expected_behavior": f"behavior {i}",
        "difficulty": "easy",
        "category": "general",
    }


# --- parse_test_cases ---------------------------------------------------


def test_strict_json_list():
    text = json.dumps([_case(1), _case(2)])
    out = parse_test_cases(text)
    assert len(out) == 2
    assert out[0]["task_input"] == "input 1"


def test_fenced_json():
    text = "```json\n" + json.dumps([_case(1)]) + "\n```"
    out = parse_test_cases(text)
    assert len(out) == 1


def test_fence_without_end():
    text = "```json\n" + json.dumps([_case(1), _case(2)])
    out = parse_test_cases(text)
    assert len(out) == 2


def test_prose_before_and_after_array():
    text = "Here are the cases:\n" + json.dumps([_case(1)]) + "\nHope this helps!"
    out = parse_test_cases(text)
    assert len(out) == 1


def test_dict_wrapper_with_list():
    text = json.dumps({"test_cases": [_case(1), _case(2), _case(3)]})
    out = parse_test_cases(text)
    assert len(out) == 3


def test_trailing_comma_in_object_and_array():
    text = (
        '[\n'
        '  {"task_input": "a", "expected_behavior": "b", "difficulty": "easy",},\n'
        '  {"task_input": "c", "expected_behavior": "d"},\n'
        ']'
    )
    out = parse_test_cases(text)
    assert len(out) == 2
    assert out[0]["task_input"] == "a"


def test_comma_and_bracket_inside_string_value_stay():
    # The string value holds a comma and a closing bracket; they must survive.
    value = "do x, then y] and stop"
    text = '[{"task_input": "' + value + '", "expected_behavior": "ok"},]'
    out = parse_test_cases(text)
    assert len(out) == 1
    assert out[0]["task_input"] == value


def test_remove_trailing_commas_keeps_string_contents():
    s = '{"a": "x, y] z",}'
    assert _remove_trailing_commas(s) == '{"a": "x, y] z"}'


def test_text_cut_in_middle_of_object_4_gives_3():
    full = [_case(1), _case(2), _case(3), _case(4)]
    whole = json.dumps(full)
    # Cut partway through the fourth object: find start of 4th object.
    marker = '{"task_input": "input 4"'
    cut_at = whole.index(marker) + len(marker) + 5
    text = whole[:cut_at]
    out = parse_test_cases(text)
    assert len(out) == 3
    assert [o["task_input"] for o in out] == ["input 1", "input 2", "input 3"]


def test_empty_string_gives_empty_list():
    assert parse_test_cases("") == []


def test_none_like_text_gives_empty_list():
    assert parse_test_cases("null") == []
    assert parse_test_cases("None") == []


def test_plain_prose_gives_empty_list():
    assert parse_test_cases("I could not generate any cases, sorry.") == []


def test_never_raises_for_str():
    for bad in ["", "[", "]", "{", "[}", "not json", "```", "```json", "[{,,}]"]:
        assert isinstance(parse_test_cases(bad), list)


# --- generate with a stubbed generator ----------------------------------


class _StubGenerator:
    """Stands in for dspy.Predict: returns queued texts, records calls."""

    class _Result:
        def __init__(self, text):
            self.test_cases = text

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        # Repeat the last text if asked more times than queued.
        idx = min(len(self.calls) - 1, len(self._texts) - 1)
        return self._Result(self._texts[idx])


@pytest.fixture(autouse=True)
def _no_real_lm(monkeypatch):
    # make_lm must not reach the network during generate().
    monkeypatch.setattr(dataset_builder, "make_lm", lambda model: object())
    monkeypatch.setattr(dataset_builder.dspy, "context", lambda **kw: _nullcontext())


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _builder(stub):
    cfg = EvolutionConfig(hermes_agent_path=None)
    b = SyntheticDatasetBuilder(cfg)
    b.generator = stub
    return b


def test_generate_retries_until_good():
    good = json.dumps([_case(1), _case(2), _case(3)])
    stub = _StubGenerator(["not json", "still bad", good])
    b = _builder(stub)
    ds = b.generate("artifact", num_cases=3)
    assert len(stub.calls) == 3
    assert len(ds.all_examples) == 3


def test_generate_raises_after_three_bad():
    stub = _StubGenerator(["bad", "bad", "bad"])
    b = _builder(stub)
    with pytest.raises(ValueError):
        b.generate("artifact", num_cases=3)
    assert len(stub.calls) == MAX_GENERATION_ATTEMPTS


def test_generate_retry_varies_rollout_id():
    good = json.dumps([_case(1), _case(2), _case(3)])
    stub = _StubGenerator(["bad", good])
    b = _builder(stub)
    b.generate("artifact", num_cases=3)
    assert "config" not in stub.calls[0]
    assert stub.calls[1]["config"]["rollout_id"] == 2
    assert stub.calls[1]["config"]["temperature"] > 0


def test_exactly_three_valid_cases_split_1_1_1():
    good = json.dumps([_case(1), _case(2), _case(3)])
    stub = _StubGenerator([good])
    b = _builder(stub)
    ds = b.generate("artifact", num_cases=3)
    assert len(ds.train) == 1
    assert len(ds.val) == 1
    assert len(ds.holdout) == 1


def test_items_without_task_input_not_counted():
    # Two valid plus two missing task_input => only 2 valid => below MIN,
    # so with a single queued text it retries and finally raises.
    cases = [
        _case(1),
        {"expected_behavior": "no input"},
        _case(2),
        {"task_input": "", "expected_behavior": "empty"},
    ]
    text = json.dumps(cases)
    stub = _StubGenerator([text])
    b = _builder(stub)
    with pytest.raises(ValueError):
        b.generate("artifact", num_cases=4)
    assert len(stub.calls) == MAX_GENERATION_ATTEMPTS


def test_holdout_not_empty_with_many_cases():
    cases = [_case(i) for i in range(1, 9)]
    stub = _StubGenerator([json.dumps(cases)])
    b = _builder(stub)
    ds = b.generate("artifact", num_cases=8)
    assert len(ds.holdout) >= 1
    assert len(ds.all_examples) == 8
