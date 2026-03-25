from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from auto_llm_innovator.constants import PHASES
from auto_llm_innovator.filesystem import ensure_dir, write_json, write_text


PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class SkillSpec:
    name: str
    summary: str
    source: dict[str, str]
    pinned_ref: str
    reviewed: bool
    optional: bool
    roles: list[str]
    phases: list[str]
    trigger_conditions: list[str]
    outputs: list[str]
    guardrails: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentSkillProfile:
    agent_role: str
    always_on_skills: list[str]
    optional_skills: list[str]
    phase_scoped_skills: dict[str, list[str]]
    forbidden_skills: list[str]
    review_requirements: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SkillRegistry:
    version: int
    policy: dict[str, Any]
    skills: dict[str, SkillSpec] = field(default_factory=dict)
    agent_profiles: dict[str, AgentSkillProfile] = field(default_factory=dict)
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy": self.policy,
            "skills": [skill.to_dict() for skill in self.skills.values()],
            "agent_profiles": {name: profile.to_dict() for name, profile in self.agent_profiles.items()},
            "source_path": self.source_path,
        }


def _candidate_registry_paths(root: Path) -> list[Path]:
    return [root / "orchestration" / "skills.json", PACKAGE_PROJECT_ROOT / "orchestration" / "skills.json"]


def _existing_path(path: str, root: Path) -> Path:
    candidate = root / path
    if candidate.exists():
        return candidate
    fallback = PACKAGE_PROJECT_ROOT / path
    return fallback


def load_skill_registry(root: Path) -> SkillRegistry:
    registry_path = next((path for path in _candidate_registry_paths(root) if path.exists()), None)
    if registry_path is None:
        raise FileNotFoundError("No skill registry found under orchestration/skills.json.")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    skills = {item["name"]: SkillSpec(**item) for item in payload["skills"]}
    profiles = {name: AgentSkillProfile(**item) for name, item in payload["agent_profiles"].items()}
    return SkillRegistry(
        version=payload["version"],
        policy=payload["policy"],
        skills=skills,
        agent_profiles=profiles,
        source_path=str(registry_path),
    )


def list_skills(root: Path) -> dict[str, Any]:
    registry = load_skill_registry(root)
    return {
        "source_path": registry.source_path,
        "policy": registry.policy,
        "skills": [skill.to_dict() for skill in registry.skills.values()],
    }


def _skill_descriptor(skill: SkillSpec, root: Path, activation_reason: str) -> dict[str, Any]:
    payload = skill.to_dict()
    payload["activation_reason"] = activation_reason
    if skill.source["type"] == "internal":
        payload["resolved_path"] = str(_existing_path(skill.source["path"], root))
    return payload


def explain_skill_profile(root: Path, role: str, phase: str | None = None) -> dict[str, Any]:
    registry = load_skill_registry(root)
    if role not in registry.agent_profiles:
        raise KeyError(f"Unknown role: {role}")
    profile = registry.agent_profiles[role]
    selected_phase = phase if phase in PHASES else None

    always_on = [
        _skill_descriptor(registry.skills[name], root, "always_on")
        for name in profile.always_on_skills
        if name not in profile.forbidden_skills
    ]
    optional = [
        _skill_descriptor(registry.skills[name], root, "optional")
        for name in profile.optional_skills
        if name not in profile.forbidden_skills
    ]
    phase_skills: dict[str, list[dict[str, Any]]] = {}
    phase_iterable = (selected_phase,) if selected_phase else PHASES
    for current_phase in phase_iterable:
        phase_skills[current_phase] = [
            _skill_descriptor(registry.skills[name], root, f"phase:{current_phase}")
            for name in profile.phase_scoped_skills.get(current_phase, [])
            if name not in profile.forbidden_skills
        ]
    active = list(always_on)
    if selected_phase:
        active.extend(phase_skills[selected_phase])
    return {
        "agent_role": role,
        "phase": selected_phase,
        "always_on": always_on,
        "optional": optional,
        "phase_scoped": phase_skills,
        "active": active,
        "forbidden": profile.forbidden_skills,
        "review_requirements": profile.review_requirements,
    }


def doctor_skill_registry(root: Path) -> dict[str, Any]:
    registry = load_skill_registry(root)
    issues: list[str] = []
    warnings: list[str] = []

    for name, skill in registry.skills.items():
        if skill.source["type"] == "external":
            if not skill.reviewed:
                issues.append(f"{name}: external skill is not reviewed.")
            if not skill.pinned_ref:
                issues.append(f"{name}: missing pinned_ref.")
            if registry.policy.get("allow_live_marketplace_installs"):
                issues.append("Policy must disable live marketplace installs.")
        elif skill.source["type"] == "internal":
            path = _existing_path(skill.source["path"], root)
            if not path.exists():
                issues.append(f"{name}: missing internal skill file at {path}.")
        else:
            issues.append(f"{name}: unsupported source type {skill.source['type']}.")

    for role, profile in registry.agent_profiles.items():
        for bucket in [profile.always_on_skills, profile.optional_skills, profile.forbidden_skills]:
            for skill_name in bucket:
                if skill_name not in registry.skills:
                    issues.append(f"{role}: references unknown skill {skill_name}.")
        for phase, skills in profile.phase_scoped_skills.items():
            if phase not in PHASES:
                issues.append(f"{role}: invalid phase {phase}.")
            for skill_name in skills:
                if skill_name not in registry.skills:
                    issues.append(f"{role}: phase {phase} references unknown skill {skill_name}.")
        if "transformers" in profile.always_on_skills:
            warnings.append(f"{role}: transformers should remain optional or forbidden.")

    return {
        "source_path": registry.source_path,
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
        "policy": registry.policy,
        "counts": {
            "skills": len(registry.skills),
            "internal": sum(1 for skill in registry.skills.values() if skill.source["type"] == "internal"),
            "external": sum(1 for skill in registry.skills.values() if skill.source["type"] == "external"),
        },
    }


def sync_reviewed_skills(root: Path) -> dict[str, Any]:
    registry = load_skill_registry(root)
    installable = [
        {
            "name": skill.name,
            "url": skill.source["url"],
            "pinned_ref": skill.pinned_ref,
        }
        for skill in registry.skills.values()
        if skill.source["type"] == "external" and skill.reviewed
    ]
    payload = {
        "policy_enforced": not registry.policy.get("allow_live_marketplace_installs", True),
        "ad_hoc_install_blocked": True,
        "installable_external_skills": installable,
        "skipped_optional_skills": [skill.name for skill in registry.skills.values() if skill.optional],
    }
    target = root / "orchestration" / "synced-skills.json"
    ensure_dir(target.parent)
    write_json(target, payload)
    return payload


def build_idea_skill_snapshot(root: Path, idea_id: str) -> tuple[dict[str, Any], str]:
    registry = load_skill_registry(root)
    snapshot = {
        "idea_id": idea_id,
        "policy": registry.policy,
        "source_path": registry.source_path,
        "skills": [skill.to_dict() for skill in registry.skills.values()],
        "roles": {role: explain_skill_profile(root, role) for role in registry.agent_profiles},
    }
    lines = [
        f"# Skill Decisions: {idea_id}",
        "",
        f"- Registry source: `{registry.source_path}`",
        f"- Live marketplace installs allowed: `{registry.policy['allow_live_marketplace_installs']}`",
        "",
    ]
    for role in registry.agent_profiles:
        explanation = explain_skill_profile(root, role)
        lines.append(f"## {role}")
        lines.append("")
        lines.append(f"- Always on: {', '.join(item['name'] for item in explanation['always_on']) or 'none'}")
        lines.append(f"- Optional: {', '.join(item['name'] for item in explanation['optional']) or 'none'}")
        for phase in PHASES:
            names = ", ".join(item["name"] for item in explanation["phase_scoped"].get(phase, [])) or "none"
            lines.append(f"- {phase}: {names}")
        lines.append(f"- Forbidden: {', '.join(explanation['forbidden']) or 'none'}")
        lines.append("")
    return snapshot, "\n".join(lines).rstrip() + "\n"


def persist_idea_skill_snapshot(root: Path, idea_dir: Path, idea_id: str) -> dict[str, Any]:
    snapshot, markdown = build_idea_skill_snapshot(root, idea_id)
    write_json(idea_dir / "orchestration" / "skills.json", snapshot)
    write_text(idea_dir / "orchestration" / "skill-decisions.md", markdown)
    return snapshot
