"""Evaluation dataset generation for hermes-agent-self-evolution.

Sources:
A) Synthetic generation — LLM reads a skill/tool/prompt and generates test cases
B) SessionDB mining — extract real usage patterns and score with LLM-as-judge
C) Golden sets — hand-curated JSONL files
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import dspy
from evolution.core.lm import make_lm

from evolution.core.config import EvolutionConfig


# Up to this many model calls are made to collect enough test cases.
MAX_GENERATION_ATTEMPTS = 3
# An attempt is good when it yields at least this many valid cases, so that
# train, val and holdout each receive one after the split.
MIN_VALID_CASES = 3

_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _strip_code_fence(text: str) -> str:
    """Remove a leading markdown code fence, with or without a closing fence."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    newline = s.find("\n")
    if newline == -1:
        # Only a fence marker, no content.
        return ""
    s = s[newline + 1:].rstrip()
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _as_case_list(obj) -> Optional[list]:
    """Coerce a decoded JSON value to a list of cases.

    Accept a list directly. Accept a dict that wraps exactly one list value,
    for example ``{"test_cases": [...]}``.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        list_values = [v for v in obj.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
    return None


def _remove_trailing_commas(s: str) -> str:
    """Drop a comma that sits before a closing ``}`` or ``]``.

    A comma inside a JSON string value is kept, so a comma or bracket that is
    part of a string stays unchanged.
    """
    out: list[str] = []
    in_string = False
    escape = False
    n = len(s)
    for i, ch in enumerate(s):
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if ch == ",":
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in "}]":
                continue
            out.append(ch)
            continue
        out.append(ch)
    return "".join(out)


def _salvage_objects(s: str) -> list:
    """Decode top-level ``{...}`` objects from the first ``[`` onward.

    Stop at the first object that does not decode (for example text that was
    cut at the output limit) and keep the objects decoded so far.
    """
    start = s.find("[")
    if start == -1:
        return []
    decoder = json.JSONDecoder()
    idx = start + 1
    n = len(s)
    results: list = []
    while idx < n:
        while idx < n and s[idx] in " \t\r\n,":
            idx += 1
        if idx >= n or s[idx] == "]":
            break
        if s[idx] != "{":
            break
        try:
            obj, end = decoder.raw_decode(s, idx)
        except json.JSONDecodeError:
            break
        results.append(obj)
        idx = end
    return results


def _parse_with_step(text: str) -> tuple[list[dict], str]:
    """Parse model test-case text and report which step succeeded.

    Steps run in sequence; the first that yields one or more dicts wins.
    Returns ``([], "none")`` when nothing parses. Never raises for ``str``.
    """
    if not isinstance(text, str):
        return [], "none"

    stripped = _strip_code_fence(text)

    def dicts_of(cand) -> list[dict]:
        if not cand:
            return []
        return [c for c in cand if isinstance(c, dict)]

    # b. Full-text JSON (list, or dict wrapping one list).
    try:
        result = dicts_of(_as_case_list(json.loads(stripped)))
        if result:
            return result, "full"
    except (json.JSONDecodeError, ValueError):
        pass

    first = stripped.find("[")
    last = stripped.rfind("]")
    if first != -1 and last != -1 and last > first:
        part = stripped[first:last + 1]
        # c. JSON of the first ``[`` to the last ``]``.
        try:
            result = dicts_of(_as_case_list(json.loads(part)))
            if result:
                return result, "bracket"
        except (json.JSONDecodeError, ValueError):
            pass
        # d. Same part after removing trailing commas.
        try:
            result = dicts_of(_as_case_list(json.loads(_remove_trailing_commas(part))))
            if result:
                return result, "repair"
        except (json.JSONDecodeError, ValueError):
            pass

    # e. Salvage whole objects until the first one that does not decode.
    result = dicts_of(_salvage_objects(stripped))
    if result:
        return result, "salvage"

    return [], "none"


def parse_test_cases(text: str) -> list[dict]:
    """Parse model output into a list of test-case dicts.

    Tolerant of fenced code, surrounding prose, a dict wrapper, trailing
    commas, and output that was cut mid-array. Returns ``[]`` when no dict can
    be parsed. Never raises for a ``str`` input.
    """
    return _parse_with_step(text)[0]


@dataclass
class EvalExample:
    """A single evaluation example."""
    task_input: str  # What the user asks
    expected_behavior: str  # Rubric — what a good response looks like
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # Category for stratified eval
    source: str = "synthetic"  # synthetic, sessiondb, golden

    def to_dict(self) -> dict:
        return {
            "task_input": self.task_input,
            "expected_behavior": self.expected_behavior,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalDataset:
    """Train/val/holdout split of evaluation examples."""
    train: list[EvalExample] = field(default_factory=list)
    val: list[EvalExample] = field(default_factory=list)
    holdout: list[EvalExample] = field(default_factory=list)

    @property
    def all_examples(self) -> list[EvalExample]:
        return self.train + self.val + self.holdout

    def save(self, path: Path):
        """Save dataset splits to JSONL files."""
        path.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in [("train", self.train), ("val", self.val), ("holdout", self.holdout)]:
            with open(path / f"{split_name}.jsonl", "w") as f:
                for ex in split_data:
                    f.write(json.dumps(ex.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        """Load dataset splits from JSONL files."""
        dataset = cls()
        for split_name in ["train", "val", "holdout"]:
            split_file = path / f"{split_name}.jsonl"
            if split_file.exists():
                examples = []
                with open(split_file) as f:
                    for line in f:
                        if line.strip():
                            examples.append(EvalExample.from_dict(json.loads(line)))
                setattr(dataset, split_name, examples)
        return dataset

    def to_dspy_examples(self, split: str = "train") -> list[dspy.Example]:
        """Convert a split to DSPy Example objects."""
        data = getattr(self, split)
        return [
            dspy.Example(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
            ).with_inputs("task_input")
            for ex in data
        ]


class SyntheticDatasetBuilder:
    """Generate evaluation datasets using a strong LLM.

    Reads the target artifact (skill file, tool description, etc.)
    and generates realistic (task_input, expected_behavior) pairs.
    """

    class GenerateTestCases(dspy.Signature):
        """Generate realistic evaluation test cases for an agent skill or tool.

        Given the full text of a skill/tool description, generate diverse test cases
        that would exercise different aspects of the skill. Each test case should include:
        - A realistic task_input (what a user would actually ask)
        - An expected_behavior rubric (what a good response should contain/do, NOT exact text)
        - A difficulty level (easy, medium, hard)
        - A category (what aspect of the skill this tests)
        """
        artifact_text: str = dspy.InputField(desc="The full text of the skill/tool/prompt being tested")
        artifact_type: str = dspy.InputField(desc="Type: 'skill', 'tool_description', or 'prompt_section'")
        num_cases: int = dspy.InputField(desc="Number of test cases to generate")
        test_cases: str = dspy.OutputField(desc="JSON array of test cases, each with: task_input, expected_behavior, difficulty, category")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.Predict(self.GenerateTestCases)

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a full eval dataset with train/val/holdout splits."""

        n = num_cases or self.config.eval_dataset_size

        # Configure DSPy to use the judge model for generation
        lm = make_lm(self.config.judge_model)

        examples: list[EvalExample] = []
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            # A repeated call with identical inputs would return the cached first
            # answer. dspy 3.4.1 bypasses its caches when rollout_id changes and
            # temperature>0 (dspy/clients/lm.py: "Different values bypass DSPy's
            # caches"; "rollout_id has no effect when temperature=0"). The first
            # attempt keeps the default request; later attempts vary rollout_id.
            call_kwargs = dict(
                artifact_text=artifact_text,
                artifact_type=artifact_type,
                num_cases=n,
            )
            if attempt > 1:
                call_kwargs["config"] = dict(rollout_id=attempt, temperature=1.0)

            with dspy.context(lm=lm):
                result = self.generator(**call_kwargs)

            raw = result.test_cases if isinstance(result.test_cases, str) else str(result.test_cases)
            cases_raw, step = _parse_with_step(raw)

            examples = [
                EvalExample(
                    task_input=c.get("task_input", ""),
                    expected_behavior=c.get("expected_behavior", ""),
                    difficulty=c.get("difficulty", "medium"),
                    category=c.get("category", "general"),
                    source="synthetic",
                )
                for c in cases_raw
                if c.get("task_input") and c.get("expected_behavior")
            ]

            if len(examples) >= MIN_VALID_CASES:
                break

            print(
                f"{_YELLOW}Dataset attempt {attempt}/{MAX_GENERATION_ATTEMPTS}: "
                f"{len(examples)} valid cases, parse step {step}, raw length {len(raw)}{_RESET}"
            )
            print(f"{_YELLOW}  raw[:200]: {raw[:200]!r}{_RESET}")
            print(f"{_YELLOW}  raw[-200:]: {raw[-200:]!r}{_RESET}")
        else:
            raise ValueError(
                f"Could not get {MIN_VALID_CASES} valid test cases from the model "
                f"after {MAX_GENERATION_ATTEMPTS} attempts"
            )

        # Shuffle and split. With at least MIN_VALID_CASES (3) examples and the
        # default 0.5/0.25 ratios, train/val/holdout each get one. Guard the
        # holdout for arbitrary ratios: when 3+ examples exist it must not be
        # empty, so cap train+val so at least one example is left for holdout.
        random.shuffle(examples)
        n_total = len(examples)
        n_train = max(1, int(n_total * self.config.train_ratio))
        n_val = max(1, int(n_total * self.config.val_ratio))
        if n_total >= 3 and n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
            if n_train + n_val >= n_total:
                n_train = n_total - n_val - 1

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )


class GoldenDatasetLoader:
    """Load hand-curated evaluation datasets from JSONL files."""

    @staticmethod
    def load(path: Path) -> EvalDataset:
        """Load a golden dataset. If no splits exist, auto-split the single file."""
        if (path / "train.jsonl").exists():
            return EvalDataset.load(path)

        # Single file — auto-split
        golden_file = path if path.suffix == ".jsonl" else path / "golden.jsonl"
        if not golden_file.exists():
            raise FileNotFoundError(f"No golden dataset found at {golden_file}")

        examples = []
        with open(golden_file) as f:
            for line in f:
                if line.strip():
                    examples.append(EvalExample.from_dict(json.loads(line)))

        random.shuffle(examples)
        n = len(examples)
        n_train = max(1, int(n * 0.5))
        n_val = max(1, int(n * 0.25))

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )
