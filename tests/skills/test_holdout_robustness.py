"""Unparseable LM answers must not abort forward() or the holdout evaluation."""

import click
import dspy
import pytest
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError

from evolution.skills.evolve_skill import _evaluate_holdout
from evolution.skills.skill_module import SkillModule


def _parse_error(lm_response):
    return AdapterParseError(
        adapter_name="ChatAdapter",
        signature=SkillModule.TaskWithSkill,
        lm_response=lm_response,
        message="Expected to find output fields in the LM response: [output]",
    )


class _RaisingPredictor:
    def __init__(self, signature, exc):
        self.signature = signature
        self.exc = exc

    def __call__(self, **kwargs):
        raise self.exc


def _module_raising(exc):
    module = SkillModule("Answer accurately")
    module.predictor = _RaisingPredictor(module.predictor.signature, exc)
    return module


def test_forward_returns_raw_answer_on_parse_error():
    module = _module_raising(_parse_error("  plain text answer\n"))
    pred = module(task_input="task")
    assert pred.output == "plain text answer"
    assert pred.skill_text == "Answer accurately"


@pytest.mark.parametrize("raw", ["", "   \n", None])
def test_forward_reraises_parse_error_without_raw_answer(raw):
    module = _module_raising(_parse_error(raw))
    with pytest.raises(AdapterParseError):
        module(task_input="task")


def test_forward_does_not_swallow_other_errors():
    module = _module_raising(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        module(task_input="task")


def test_forward_parsed_answer_unchanged():
    module = SkillModule("Answer accurately")
    with dspy.context(lm=DummyLM([{"output": "parsed answer"}])):
        pred = module(task_input="task")
    assert pred.output == "parsed answer"
    assert pred.skill_text == "Answer accurately"


class _Stub:
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)

    def __call__(self, task_input):
        if task_input in self.fail_on:
            raise _parse_error("")
        return dspy.Prediction(output=f"answer {task_input}")


def _judge(example, pred):
    return 0.5 if pred.output == "answer a" else 0.8


EXAMPLES = [dspy.Example(task_input=t).with_inputs("task_input") for t in ("a", "b")]


def test_holdout_continues_when_one_baseline_example_fails():
    baseline, evolved, errors = _evaluate_holdout(
        EXAMPLES, _Stub(fail_on={"a"}), _Stub(), _judge, lm=None)
    assert baseline == [0.0, 0.8]
    assert evolved == [0.5, 0.8]
    assert errors == {"baseline": 1, "evolved": 0}


def test_holdout_raises_when_all_examples_fail_on_both_sides():
    with pytest.raises(click.ClickException, match="failed for all examples"):
        _evaluate_holdout(EXAMPLES, _Stub(fail_on={"a", "b"}), _Stub(fail_on={"a", "b"}),
                          _judge, lm=None)


def test_holdout_all_baseline_failures_alone_do_not_raise():
    baseline, evolved, errors = _evaluate_holdout(
        EXAMPLES, _Stub(fail_on={"a", "b"}), _Stub(), _judge, lm=None)
    assert baseline == [0.0, 0.0]
    assert evolved == [0.5, 0.8]
    assert errors == {"baseline": 2, "evolved": 0}
