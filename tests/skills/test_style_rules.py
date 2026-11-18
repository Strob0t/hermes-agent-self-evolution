"""Exercise the optional STE style rules for proposed skill text."""

import importlib

import dspy
import pytest
from click.testing import CliRunner

from tests.skills.test_model_configuration import provider  # noqa: F401  (pytest fixture)

module = importlib.import_module("evolution.skills.evolve_skill")

STE_SENTENCE = "Write the new instructions in ASD-STE100 Simplified Technical English."


def test_env_not_set_returns_none(monkeypatch):
    monkeypatch.delenv("EVOLUTION_STYLE_RULES_FILES", raising=False)
    assert module._build_style_instructions() is None


def test_two_files_combined_into_mandatory_section(tmp_path, monkeypatch):
    a = tmp_path / "rules.md"
    b = tmp_path / "words.md"
    a.write_text("ALPHA-RULES-TEXT", encoding="utf-8")
    b.write_text("BETA-WORDS-TEXT", encoding="utf-8")
    monkeypatch.setenv("EVOLUTION_STYLE_RULES_FILES", f"{a},{b}")
    out = module._build_style_instructions()
    assert out is not None
    assert STE_SENTENCE in out
    assert "ALPHA-RULES-TEXT" in out
    assert "BETA-WORDS-TEXT" in out


def test_missing_file_falls_back_to_default(tmp_path, monkeypatch, capsys):
    a = tmp_path / "rules.md"
    a.write_text("ALPHA-RULES-TEXT", encoding="utf-8")
    missing = tmp_path / "does-not-exist.md"
    monkeypatch.setenv("EVOLUTION_STYLE_RULES_FILES", f"{a},{missing}")
    assert module._build_style_instructions() is None
    captured = capsys.readouterr().out
    assert "Style rules file missing or unreadable" in captured


def test_total_text_cut_at_12000(tmp_path, monkeypatch):
    a = tmp_path / "big_a.md"
    b = tmp_path / "big_b.md"
    a.write_text("A" * 10000, encoding="utf-8")
    b.write_text("B" * 10000, encoding="utf-8")
    monkeypatch.setenv("EVOLUTION_STYLE_RULES_FILES", f"{a},{b}")
    out = module._build_style_instructions()
    assert out is not None
    # Only the rules body is bounded; the prefix sentence is extra.
    body = out.split("Use these rules:\n", 1)[1]
    assert len(body.replace("\n\n", "")) == module.STYLE_RULES_MAX_CHARS
    assert out.count("B") <= 2000


def _capture_gepa(monkeypatch):
    """Record GEPA kwargs and skip the expensive search."""
    seen = {}
    real_gepa = dspy.GEPA

    def fake_gepa(**kwargs):
        seen.update(kwargs)
        instance = real_gepa(**kwargs)
        monkeypatch.setattr(type(instance), "compile", lambda self, student, **kw: student)
        return instance

    monkeypatch.setattr(dspy, "GEPA", fake_gepa)
    return seen


def test_cli_gepa_gets_proposer_with_both_files(provider, tmp_path, monkeypatch):
    requests, args, root = provider
    rules = root / "rules.md"
    words = root / "words.md"
    rules.write_text("ALPHA-RULES-TEXT", encoding="utf-8")
    words.write_text("BETA-WORDS-TEXT", encoding="utf-8")
    monkeypatch.setenv("EVOLUTION_STYLE_RULES_FILES", f"{rules},{words}")
    seen = _capture_gepa(monkeypatch)
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code == 0, result.output
    proposer = seen.get("instruction_proposer")
    assert proposer is not None
    assert STE_SENTENCE in proposer.additional_instructions
    assert "ALPHA-RULES-TEXT" in proposer.additional_instructions
    assert "BETA-WORDS-TEXT" in proposer.additional_instructions
    assert "Style rules: 2 files" in result.output


def test_cli_gepa_unchanged_without_env(provider, monkeypatch):
    requests, args, root = provider
    monkeypatch.delenv("EVOLUTION_STYLE_RULES_FILES", raising=False)
    seen = _capture_gepa(monkeypatch)
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code == 0, result.output
    assert "instruction_proposer" not in seen
    assert "Style rules:" not in result.output


def test_cli_missing_file_uses_default_and_warns(provider, monkeypatch):
    requests, args, root = provider
    rules = root / "rules.md"
    rules.write_text("ALPHA-RULES-TEXT", encoding="utf-8")
    monkeypatch.setenv("EVOLUTION_STYLE_RULES_FILES", f"{rules},{root / 'nope.md'}")
    seen = _capture_gepa(monkeypatch)
    result = CliRunner().invoke(module.main, args)
    assert result.exit_code == 0, result.output
    assert "instruction_proposer" not in seen
    assert "Style rules file missing or unreadable" in result.output
