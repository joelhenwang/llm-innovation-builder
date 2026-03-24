from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from auto_llm_innovator.filesystem import append_jsonl, read_json, write_json
from auto_llm_innovator.modeling.interfaces import PhaseResult


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def create_attempt_record(idea_dir: Path, attempt_id: str, parent_attempt: str | None = None) -> dict:
    ledger_path = idea_dir / "ledger.jsonl"
    payload = {
        "event": "attempt_created",
        "attempt_id": attempt_id,
        "parent_attempt": parent_attempt,
        "timestamp": utc_now(),
    }
    append_jsonl(ledger_path, payload)
    status_path = idea_dir / "status.json"
    status = read_json(status_path) if status_path.exists() else {"idea_id": idea_dir.name, "attempts": []}
    status["attempts"].append(
        {
            "attempt_id": attempt_id,
            "parent_attempt": parent_attempt,
            "phases": {},
            "created_at": payload["timestamp"],
            "state": "running",
        }
    )
    write_json(status_path, status)
    return payload


def record_phase_result(idea_dir: Path, result: PhaseResult) -> None:
    append_jsonl(
        idea_dir / "ledger.jsonl",
        {
            "event": "phase_result",
            "timestamp": utc_now(),
            **result.to_dict(),
        },
    )
    status = read_json(idea_dir / "status.json")
    for attempt in status["attempts"]:
        if attempt["attempt_id"] == result.attempt_id:
            attempt["phases"][result.phase] = result.to_dict()
            attempt["state"] = "completed" if result.phase == "full" and result.status == "passed" else "running"
            break
    write_json(idea_dir / "status.json", status)


def finalize_attempt(idea_dir: Path, attempt_id: str, final_state: str) -> None:
    status = read_json(idea_dir / "status.json")
    for attempt in status["attempts"]:
        if attempt["attempt_id"] == attempt_id:
            attempt["state"] = final_state
            attempt["completed_at"] = utc_now()
            break
    write_json(idea_dir / "status.json", status)


def load_status(idea_dir: Path) -> dict:
    return read_json(idea_dir / "status.json")
