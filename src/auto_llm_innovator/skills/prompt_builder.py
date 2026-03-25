from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

from auto_llm_innovator.idea_spec.models import IdeaSpec
from auto_llm_innovator.skills.registry import explain_skill_profile, load_skill_registry


DEFAULT_PROMPT_CONTEXT = {
    "needs_capability_search": False,
    "needs_tokenizer_api_compatibility": False,
    "external_tracking_enabled": False,
    "curriculum_redesign": False,
    "non_default_eval_suite": False,
    "use_lightning": False,
    "skill_development_mode": False,
}


@dataclass(slots=True)
class PromptBuildResult:
    role: str
    phase: str
    system_prompt: str
    user_prompt: str
    active_skills: list[dict[str, Any]]
    injected_skills: list[dict[str, Any]]
    skipped_skills: list[dict[str, Any]]
    skill_snippets: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return text
    return parts[1]


def _extract_markdown_section_bullets(text: str) -> dict[str, list[str]]:
    body = _strip_frontmatter(text)
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections.setdefault(current, [])
            continue
        if current is None or not line:
            continue
        if re.match(r"^[-*]\s+", line):
            sections[current].append(re.sub(r"^[-*]\s+", "", line))
        elif re.match(r"^\d+\.\s+", line):
            sections[current].append(re.sub(r"^\d+\.\s+", "", line))
    return sections


def _skill_directives(skill: dict[str, Any], root: Path) -> list[str]:
    directives = [skill["summary"]]
    trigger_conditions = skill.get("trigger_conditions", [])
    if trigger_conditions:
        directives.append(f"When to use: {trigger_conditions[0]}")

    if skill["source"]["type"] == "internal" and skill.get("resolved_path"):
        path = Path(skill["resolved_path"])
        if path.exists():
            sections = _extract_markdown_section_bullets(path.read_text(encoding="utf-8"))
            for item in sections.get("workflow", [])[:2]:
                directives.append(f"Workflow: {item}")
            for item in sections.get("required output", [])[:2]:
                directives.append(f"Output: {item}")
            for item in sections.get("guardrails", [])[:2]:
                directives.append(f"Guardrail: {item}")
    else:
        for item in skill.get("outputs", [])[:1]:
            directives.append(f"Output: {item}")
        for item in skill.get("guardrails", [])[:2]:
            directives.append(f"Guardrail: {item}")

    deduped: list[str] = []
    for item in directives:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _optional_skill_enabled(skill_name: str, context: dict[str, bool], role: str) -> tuple[bool, str | None]:
    conditions = {
        "find-skills": ("needs_capability_search", "Only inject when the planner is explicitly searching for missing capabilities."),
        "transformers": (
            "needs_tokenizer_api_compatibility",
            "Only inject for tokenizer/API compatibility questions to avoid baseline drift.",
        ),
        "weights-and-biases": ("external_tracking_enabled", "Only inject when external tracking is enabled."),
        "dataset-curriculum-designer": ("curriculum_redesign", "Only inject during active curriculum redesign."),
        "eval-suite-builder": ("non_default_eval_suite", "Only inject when a non-default evaluation suite is required."),
        "pytorch-lightning": ("use_lightning", "Only inject if the framework intentionally adopts Lightning."),
        "skill-creator": ("skill_development_mode", "Only inject during skill authoring or revision."),
    }
    if skill_name not in conditions:
        return False, None
    flag, reason = conditions[skill_name]
    return bool(context.get(flag, False)), reason


def _should_inject_active_skill(skill_name: str, context: dict[str, bool], role: str) -> tuple[bool, str | None]:
    if skill_name in {"find-skills", "pytorch-lightning", "skill-creator"}:
        enabled, reason = _optional_skill_enabled(skill_name, context, role)
        return enabled, reason
    return True, None


def _dedupe_skills(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for skill in skills:
        if skill["name"] in seen:
            continue
        seen.add(skill["name"])
        deduped.append(skill)
    return deduped


def build_agent_prompt(
    spec: IdeaSpec,
    role: str,
    phase: str,
    root: Path,
    context: dict[str, bool] | None = None,
) -> PromptBuildResult:
    from auto_llm_innovator.orchestration.agents import agent_definitions

    merged_context = DEFAULT_PROMPT_CONTEXT.copy()
    if context:
        merged_context.update(context)

    registry = load_skill_registry(root)
    if role not in registry.agent_profiles:
        raise KeyError(f"Unknown role: {role}")

    base = agent_definitions(spec, root=root)[role]
    explanation = explain_skill_profile(root, role, phase=phase)
    active = list(explanation["active"])
    for skill in explanation["optional"]:
        enabled, _ = _optional_skill_enabled(skill["name"], merged_context, role)
        if enabled:
            active.append(skill)
    active = _dedupe_skills(active)

    injected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    snippets: dict[str, list[str]] = {}

    for skill in active:
        should_inject, reason = _should_inject_active_skill(skill["name"], merged_context, role)
        if not should_inject:
            skipped.append(
                {
                    "name": skill["name"],
                    "reason": reason or "Not prompt-worthy under the current context.",
                    "activation_reason": skill.get("activation_reason"),
                }
            )
            continue
        snippets[skill["name"]] = _skill_directives(skill, root)
        injected.append(skill)

    system_lines = [str(base["system_prompt"])]
    if injected:
        system_lines.extend(["", "Injected skill directives:"])
        for skill in injected:
            system_lines.append(f"- {skill['name']}: {snippets[skill['name']][0]}")
            for detail in snippets[skill["name"]][1:3]:
                system_lines.append(f"  {detail}")
    if base["forbidden_skills"]:
        system_lines.extend(["", f"Forbidden skills: {', '.join(base['forbidden_skills'])}"])
    if base["review_requirements"]:
        system_lines.extend(["", "Review requirements:"])
        for requirement in base["review_requirements"]:
            system_lines.append(f"- {requirement}")

    user_lines = [
        f"Idea ID: {spec.idea_id}",
        f"Role: {role}",
        f"Phase: {phase}",
        f"Goal: {base['goal']}",
        "",
        "Brief:",
        spec.raw_brief,
        "",
        f"Hypothesis: {spec.hypothesis}",
        f"Target model: {spec.intended_model_target}",
        f"Tokenizer: {spec.tokenizer}",
        "",
        "Novelty claims:",
    ]
    user_lines.extend(f"- {claim}" for claim in spec.novelty_claims)
    user_lines.extend(["", "Forbidden fallback patterns:"])
    user_lines.extend(f"- {pattern}" for pattern in spec.forbidden_fallback_patterns)
    user_lines.extend(["", "Training curriculum outline:"])
    user_lines.extend(f"- {item}" for item in spec.training_curriculum_outline)
    if injected:
        user_lines.extend(["", "Use the injected skill directives above as compact operating guidance."])

    return PromptBuildResult(
        role=role,
        phase=phase,
        system_prompt="\n".join(system_lines).strip() + "\n",
        user_prompt="\n".join(user_lines).strip() + "\n",
        active_skills=active,
        injected_skills=injected,
        skipped_skills=skipped,
        skill_snippets=snippets,
    )
