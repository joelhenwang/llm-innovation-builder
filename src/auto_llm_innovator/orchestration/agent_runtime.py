from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from auto_llm_innovator.filesystem import ensure_dir, write_json
from auto_llm_innovator.tracking.lineage import collect_artifact_record


VALID_PARSE_STATUSES = {"valid", "invalid_json", "invalid_schema", "dry_run", "runtime_failed"}
VALID_REVIEWER_RECOMMENDATIONS = {"promote", "continue_with_caution", "rerun_with_more_budget", "stop"}


@dataclass(slots=True)
class AgentContextArtifactRef:
    path: str
    kind: str
    sha256: str | None = None
    missing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRequestEnvelope:
    idea_id: str
    attempt_id: str
    phase: str
    role: str
    expected_response_kind: str
    prompt_payload: dict[str, Any]
    context_artifacts: list[AgentContextArtifactRef]
    response_format_instructions: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["context_artifacts"] = [artifact.to_dict() for artifact in self.context_artifacts]
        return payload


@dataclass(slots=True)
class AgentInvocationRecord:
    role: str
    phase: str
    status: str
    parse_status: str
    validation_errors: list[str]
    command_preview: dict[str, Any]
    raw_stdout: str = ""
    raw_stderr: str = ""
    parsed_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlannerResponse:
    phase_summary: str
    focus_areas: list[str]
    risk_flags: list[str]
    success_criteria: list[str]
    recommended_next_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewerResponse:
    recommendation: str
    summary: str
    blocking_concerns: list[str]
    continuation_criteria: list[str]
    cited_artifacts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def artifact_ref_for_path(path: Path, *, kind: str, relative_to: Path) -> AgentContextArtifactRef:
    record = collect_artifact_record(path, kind=kind, relative_to=relative_to)
    return AgentContextArtifactRef(
        path=str(record["path"]),
        kind=str(record["kind"]),
        sha256=str(record["sha256"]) if record.get("sha256") else None,
        missing=bool(record.get("missing", False)),
    )


def build_agent_request_envelope(
    *,
    idea_id: str,
    attempt_id: str,
    phase: str,
    role: str,
    expected_response_kind: str,
    prompt_payload: dict[str, Any],
    context_artifacts: list[AgentContextArtifactRef],
    context: dict[str, Any] | None = None,
) -> AgentRequestEnvelope:
    return AgentRequestEnvelope(
        idea_id=idea_id,
        attempt_id=attempt_id,
        phase=phase,
        role=role,
        expected_response_kind=expected_response_kind,
        prompt_payload=prompt_payload,
        context_artifacts=context_artifacts,
        response_format_instructions=_response_format_instructions(role, context_artifacts),
        context=dict(context or {}),
    )


def validate_agent_payload(
    *,
    role: str,
    payload: dict[str, Any] | None,
    allowed_artifact_paths: set[str],
) -> tuple[str, dict[str, Any] | None, list[str]]:
    if payload is None:
        return "invalid_json", None, ["No JSON payload was produced."]
    if not isinstance(payload, dict):
        return "invalid_schema", None, ["Structured payload must be a JSON object."]

    if role == "planner":
        return _validate_planner_payload(payload)
    if role == "reviewer":
        return _validate_reviewer_payload(payload, allowed_artifact_paths=allowed_artifact_paths)
    return "invalid_schema", None, [f"Unsupported structured role: {role}"]


def build_agent_response_artifact(
    *,
    request: AgentRequestEnvelope,
    parse_status: str,
    validation_errors: list[str],
    normalized_payload: dict[str, Any] | None,
    raw_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "idea_id": request.idea_id,
        "attempt_id": request.attempt_id,
        "phase": request.phase,
        "role": request.role,
        "response_kind": request.expected_response_kind,
        "parse_status": parse_status,
        "validation_errors": list(validation_errors),
        "payload": normalized_payload,
        "raw_payload": raw_payload if parse_status != "valid" else None,
    }


def persist_agent_request(phase_dir: Path, request: AgentRequestEnvelope) -> Path:
    agents_dir = ensure_dir(phase_dir / "agents")
    path = agents_dir / f"{request.role}-request.json"
    write_json(path, request.to_dict())
    return path


def persist_agent_response(phase_dir: Path, role: str, payload: dict[str, Any]) -> Path:
    agents_dir = ensure_dir(phase_dir / "agents")
    path = agents_dir / f"{role}-response.json"
    write_json(path, payload)
    return path


def persist_agent_runtime(phase_dir: Path, record: AgentInvocationRecord) -> Path:
    agents_dir = ensure_dir(phase_dir / "agents")
    path = agents_dir / f"{record.role}-runtime.json"
    write_json(path, record.to_dict())
    return path


def render_structured_prompt(request: AgentRequestEnvelope) -> str:
    system_prompt = str(request.prompt_payload.get("system_prompt", "")).rstrip()
    user_prompt = str(request.prompt_payload.get("user_prompt", "")).rstrip()
    artifact_lines = [
        f"- {artifact.kind}: {artifact.path}" + (f" (sha256={artifact.sha256})" if artifact.sha256 else "")
        for artifact in request.context_artifacts
    ]
    lines = [
        system_prompt,
        "",
        request.response_format_instructions,
        "",
        user_prompt,
    ]
    if artifact_lines:
        lines.extend(["", "Referenced artifacts:"])
        lines.extend(artifact_lines)
    if request.context:
        lines.extend(["", "Structured context:", str(request.context)])
    return "\n".join(line for line in lines if line is not None).strip() + "\n"


def _response_format_instructions(role: str, context_artifacts: list[AgentContextArtifactRef]) -> str:
    if role == "planner":
        return (
            "Return only a JSON object with keys: "
            "`phase_summary` (string), `focus_areas` (array of strings), `risk_flags` (array of strings), "
            "`success_criteria` (array of strings), `recommended_next_action` (string)."
        )
    allowed_paths = [artifact.path for artifact in context_artifacts if not artifact.missing]
    return (
        "Return only a JSON object with keys: "
        "`recommendation` (one of: promote, continue_with_caution, rerun_with_more_budget, stop), "
        "`summary` (string), `blocking_concerns` (array of strings), `continuation_criteria` (array of strings), "
        "`cited_artifacts` (array of strings chosen only from these paths: "
        + ", ".join(allowed_paths)
        + ")."
    )


def _validate_planner_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    phase_summary = _require_non_empty_string(payload, "phase_summary", errors)
    focus_areas = _require_string_list(payload, "focus_areas", errors)
    risk_flags = _require_string_list(payload, "risk_flags", errors)
    success_criteria = _require_string_list(payload, "success_criteria", errors)
    recommended_next_action = _require_non_empty_string(payload, "recommended_next_action", errors)
    if errors:
        return "invalid_schema", None, errors
    response = PlannerResponse(
        phase_summary=phase_summary,
        focus_areas=focus_areas,
        risk_flags=risk_flags,
        success_criteria=success_criteria,
        recommended_next_action=recommended_next_action,
    )
    return "valid", response.to_dict(), []


def _validate_reviewer_payload(
    payload: dict[str, Any],
    *,
    allowed_artifact_paths: set[str],
) -> tuple[str, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    recommendation = _require_non_empty_string(payload, "recommendation", errors)
    if recommendation and recommendation not in VALID_REVIEWER_RECOMMENDATIONS:
        errors.append(
            "recommendation must be one of: "
            + ", ".join(sorted(VALID_REVIEWER_RECOMMENDATIONS))
        )
    summary = _require_non_empty_string(payload, "summary", errors)
    blocking_concerns = _require_string_list(payload, "blocking_concerns", errors)
    continuation_criteria = _require_string_list(payload, "continuation_criteria", errors)
    cited_artifacts = _require_string_list(payload, "cited_artifacts", errors)
    invalid_paths = [path for path in cited_artifacts if path not in allowed_artifact_paths]
    if invalid_paths:
        errors.append(f"cited_artifacts contains unknown paths: {', '.join(invalid_paths)}")
    if errors:
        return "invalid_schema", None, errors
    response = ReviewerResponse(
        recommendation=recommendation,
        summary=summary,
        blocking_concerns=blocking_concerns,
        continuation_criteria=continuation_criteria,
        cited_artifacts=cited_artifacts,
    )
    return "valid", response.to_dict(), []


def _require_non_empty_string(payload: dict[str, Any], key: str, errors: list[str]) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string.")
        return ""
    return value.strip()


def _require_string_list(payload: dict[str, Any], key: str, errors: list[str]) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{key} must be a list of non-empty strings.")
        return []
    return [item.strip() for item in value]
