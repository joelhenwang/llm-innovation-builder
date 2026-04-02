from types import SimpleNamespace
from pathlib import Path

from auto_llm_innovator.orchestration.opencode import OpenCodeAdapter


def test_invoke_structured_returns_dry_run_when_unavailable(tmp_path: Path):
    adapter = OpenCodeAdapter(executable="definitely-missing-opencode")

    payload = adapter.invoke_structured(
        role="planner",
        system_prompt="system",
        user_prompt="user",
        response_format_instructions="Return JSON.",
        cwd=tmp_path,
    )

    assert payload["status"] == "dry-run"
    assert payload["parse_status"] == "dry_run"
    assert payload["parsed_payload"] is None


def test_invoke_structured_parses_fenced_json(monkeypatch, tmp_path: Path):
    adapter = OpenCodeAdapter()
    monkeypatch.setattr(OpenCodeAdapter, "available", lambda self: True)
    monkeypatch.setattr(
        "auto_llm_innovator.orchestration.opencode.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='```json\n{"phase_summary":"x","focus_areas":["a"],"risk_flags":[],"success_criteria":["b"],"recommended_next_action":"execute"}\n```',
            stderr="",
        ),
    )

    payload = adapter.invoke_structured(
        role="planner",
        system_prompt="system",
        user_prompt="user",
        response_format_instructions="Return JSON.",
        cwd=tmp_path,
    )

    assert payload["status"] == "completed"
    assert payload["parse_status"] == "valid"
    assert payload["parsed_payload"]["phase_summary"] == "x"


def test_invoke_structured_marks_runtime_failed_on_nonzero_exit(monkeypatch, tmp_path: Path):
    adapter = OpenCodeAdapter()
    monkeypatch.setattr(OpenCodeAdapter, "available", lambda self: True)
    monkeypatch.setattr(
        "auto_llm_innovator.orchestration.opencode.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout="", stderr="boom"),
    )

    payload = adapter.invoke_structured(
        role="reviewer",
        system_prompt="system",
        user_prompt="user",
        response_format_instructions="Return JSON.",
        cwd=tmp_path,
    )

    assert payload["status"] == "failed"
    assert payload["parse_status"] == "runtime_failed"
    assert payload["validation_errors"]
