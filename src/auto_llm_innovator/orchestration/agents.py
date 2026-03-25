from __future__ import annotations

from pathlib import Path

from auto_llm_innovator.idea_spec.models import IdeaSpec
from auto_llm_innovator.skills.registry import explain_skill_profile


def agent_definitions(spec: IdeaSpec, root: Path) -> dict[str, dict[str, object]]:
    common = (
        "You are part of an autonomous LLM innovator framework. "
        "Do not fallback to a generic transformer template. "
        "Study inspirations critically, declare borrowed mechanisms, and preserve originality."
    )
    planner_skills = explain_skill_profile(root, "planner")
    implementer_skills = explain_skill_profile(root, "implementer")
    debugger_skills = explain_skill_profile(root, "debugger")
    trainer_skills = explain_skill_profile(root, "trainer")
    evaluator_skills = explain_skill_profile(root, "evaluator")
    reviewer_skills = explain_skill_profile(root, "reviewer")
    return {
        "planner": {
            "goal": "Translate the idea brief into an implementation and experiment plan.",
            "system_prompt": f"{common} Produce a phase-aware plan for idea {spec.idea_id}.",
            "mandatory_skills": [item["name"] for item in planner_skills["always_on"]],
            "optional_skills": [item["name"] for item in planner_skills["optional"]],
            "phase_scoped_skills": {phase: [item["name"] for item in items] for phase, items in planner_skills["phase_scoped"].items()},
            "forbidden_skills": planner_skills["forbidden"],
            "review_requirements": planner_skills["review_requirements"],
        },
        "implementer": {
            "goal": "Author model code, configs, and wiring for the idea-specific plugin bundle.",
            "system_prompt": f"{common} Build original modeling code for {spec.idea_id} under the 2.1B cap.",
            "mandatory_skills": [item["name"] for item in implementer_skills["always_on"]],
            "optional_skills": [item["name"] for item in implementer_skills["optional"]],
            "phase_scoped_skills": {phase: [item["name"] for item in items] for phase, items in implementer_skills["phase_scoped"].items()},
            "forbidden_skills": implementer_skills["forbidden"],
            "review_requirements": implementer_skills["review_requirements"],
        },
        "debugger": {
            "goal": "Investigate tensor, logic, math, or training failures and propose revisions.",
            "system_prompt": f"{common} Prioritize smoke-test correctness and explain root causes clearly.",
            "mandatory_skills": [item["name"] for item in debugger_skills["always_on"]],
            "optional_skills": [item["name"] for item in debugger_skills["optional"]],
            "phase_scoped_skills": {phase: [item["name"] for item in items] for phase, items in debugger_skills["phase_scoped"].items()},
            "forbidden_skills": debugger_skills["forbidden"],
            "review_requirements": debugger_skills["review_requirements"],
        },
        "trainer": {
            "goal": "Run phase-specific training and record budget, metrics, and anomalies.",
            "system_prompt": f"{common} Use GPT-2 tokenization and respect visible retry and budget metadata.",
            "mandatory_skills": [item["name"] for item in trainer_skills["always_on"]],
            "optional_skills": [item["name"] for item in trainer_skills["optional"]],
            "phase_scoped_skills": {phase: [item["name"] for item in items] for phase, items in trainer_skills["phase_scoped"].items()},
            "forbidden_skills": trainer_skills["forbidden"],
            "review_requirements": trainer_skills["review_requirements"],
        },
        "evaluator": {
            "goal": "Compare results against baseline, prior runs, and public references.",
            "system_prompt": f"{common} Focus on learnability signals, originality, and comparative metrics.",
            "mandatory_skills": [item["name"] for item in evaluator_skills["always_on"]],
            "optional_skills": [item["name"] for item in evaluator_skills["optional"]],
            "phase_scoped_skills": {phase: [item["name"] for item in items] for phase, items in evaluator_skills["phase_scoped"].items()},
            "forbidden_skills": evaluator_skills["forbidden"],
            "review_requirements": evaluator_skills["review_requirements"],
        },
        "reviewer": {
            "goal": "Gate originality and decide whether to continue, redesign, or stop.",
            "system_prompt": f"{common} Reject near-copy architectures and require written rationale for advancement.",
            "mandatory_skills": [item["name"] for item in reviewer_skills["always_on"]],
            "optional_skills": [item["name"] for item in reviewer_skills["optional"]],
            "phase_scoped_skills": {phase: [item["name"] for item in items] for phase, items in reviewer_skills["phase_scoped"].items()},
            "forbidden_skills": reviewer_skills["forbidden"],
            "review_requirements": reviewer_skills["review_requirements"],
        },
    }
