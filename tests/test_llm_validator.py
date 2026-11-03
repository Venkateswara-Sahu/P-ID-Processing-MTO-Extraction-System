"""
LLM validator tests.

Covers the graceful-degradation paths, the retired-model bug, and the
response parsing that mis-detected pass/fail.
"""
import pytest

from src.validation.llm_validator import LLMValidator


# --------------------------------------------------------------------------- #
# C1: the configured model must be one Groq actually serves
# --------------------------------------------------------------------------- #

def test_default_model_is_not_retired():
    """
    'llama-3.3-70b-versatile' returns HTTP 404 from Groq, so LLM validation
    never ran while appearing to succeed.
    """
    from config.settings import settings

    assert settings.groq_model != "llama-3.3-70b-versatile"


def test_no_api_key_degrades_gracefully():
    """A missing key must not raise; it must report that it did not run."""
    validator = LLMValidator(api_key="")
    result = validator.validate([], {}, [])

    assert result["degraded"] is True
    assert "did not run" in result["report"].lower()
    assert result["llm_issues"] == []


def test_degraded_result_is_flagged():
    """
    The fallback must be visibly marked as degraded. A report that reads
    like a clean validation result is worse than no report.
    """
    validator = LLMValidator(api_key="")
    result = validator.validate([], {}, [{"severity": "error", "message": "x"}])
    assert result.get("degraded") is True


def test_fallback_report_does_not_blame_a_missing_key_when_one_is_set():
    """
    The key was present and valid; the failure was the model name. The
    message must not send the user to fix the wrong thing.
    """
    validator = LLMValidator(api_key="gsk_valid_looking_key")
    result = validator.validate([], {}, [])

    if result.get("degraded"):
        report = result["report"].lower()
        assert "api key" not in report or "invalid" in report, (
            "the report blames the API key even though one is configured"
        )


# --------------------------------------------------------------------------- #
# C14: a broken langgraph must not kill the pipeline
# --------------------------------------------------------------------------- #

def test_build_graph_failure_is_contained(monkeypatch):
    """
    _build_graph() was called OUTSIDE the try/except, so an ImportError
    from langgraph propagated out of validate() and killed the pipeline.
    """
    validator = LLMValidator(api_key="gsk_fake")

    def boom():
        raise ImportError("No module named 'langgraph'")

    monkeypatch.setattr(validator, "_init_llm", lambda: True)
    monkeypatch.setattr(validator, "_build_graph", boom)

    # Must not raise.
    result = validator.validate([], {}, [])
    assert result["degraded"] is True


def test_missing_langgraph_does_not_raise(monkeypatch):
    """Even a totally broken environment must degrade, not explode."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("langgraph") or name.startswith("langchain_groq"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    validator = LLMValidator(api_key="gsk_fake")
    result = validator.validate([], {}, [])
    assert result["degraded"] is True


# --------------------------------------------------------------------------- #
# S15: response parsing
# --------------------------------------------------------------------------- #

def test_issue_parser_splits_markdown_bold_headers():
    """
    '**1. Missing tag**' starts with '*', so the header was appended to the
    previous issue instead of starting a new one.
    """
    validator = LLMValidator(api_key="gsk_fake")
    raw = (
        "**1. Valve HV-1001 has no tag**\n"
        "Severity: warning\n"
        "**2. Loop 101 has no control element**\n"
        "Severity: error\n"
    )
    issues = validator._parse_issues(raw)
    assert len(issues) == 2, f"expected 2 issues, got {len(issues)}"
    assert "HV-1001" in issues[0]
    assert "Loop 101" in issues[1]


def test_issue_parser_does_not_promote_sub_bullets():
    """'- Severity: error' is a continuation, not a new issue."""
    validator = LLMValidator(api_key="gsk_fake")
    raw = (
        "1. Missing tag on valve\n"
        "- Severity: warning\n"
        "- Action: re-scan OCR region\n"
        "2. Missing control element\n"
        "- Severity: error\n"
    )
    issues = validator._parse_issues(raw)
    assert len(issues) == 2, f"expected 2 issues, got {len(issues)}"


def test_passed_is_not_computed_by_substring_matching():
    """
    'No errors detected' contains the word 'errors' and was read as a
    failure. The verdict must come from the parsed severity.
    """
    validator = LLMValidator(api_key="gsk_fake")
    raw = "1. No errors detected in the extraction.\n"
    issues = validator._parse_issues(raw)
    passed = validator._compute_passed(issues)
    assert passed is True, (
        "a reassuring sentence was misread as a validation failure"
    )


def test_a_real_error_fails():
    validator = LLMValidator(api_key="gsk_fake")
    raw = "1. Valve has no tag\nSeverity: error\n"
    issues = validator._parse_issues(raw)
    assert validator._compute_passed(issues) is False


def test_issue_parser_returns_empty_for_empty_input():
    validator = LLMValidator(api_key="gsk_fake")
    assert validator._parse_issues("") == []
