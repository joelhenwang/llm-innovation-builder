from __future__ import annotations

from auto_llm_innovator.idea_spec.models import IdeaSpec


def agent_definitions(spec: IdeaSpec) -> dict[str, dict[str, str]]:
    common = (
        "You are part of an autonomous LLM innovator framework. "
        "Do not fallback to a generic transformer template. "
        "Study inspirations critically, declare borrowed mechanisms, and preserve originality."
    )
    return {
        "planner": {
            "goal": "Translate the idea brief into an implementation and experiment plan.",
            "system_prompt": f"{common} Produce a phase-aware plan for idea {spec.idea_id}.",
        },
        "implementer": {
            "goal": "Author model code, configs, and wiring for the idea-specific plugin bundle.",
            "system_prompt": f"{common} Build original modeling code for {spec.idea_id} under the 2.1B cap.",
        },
        "debugger": {
            "goal": "Investigate tensor, logic, math, or training failures and propose revisions.",
            "system_prompt": f"{common} Prioritize smoke-test correctness and explain root causes clearly.",
        },
        "trainer": {
            "goal": "Run phase-specific training and record budget, metrics, and anomalies.",
            "system_prompt": f"{common} Use GPT-2 tokenization and respect visible retry and budget metadata.",
        },
        "evaluator": {
            "goal": "Compare results against baseline, prior runs, and public references.",
            "system_prompt": f"{common} Focus on learnability signals, originality, and comparative metrics.",
        },
        "reviewer": {
            "goal": "Gate originality and decide whether to continue, redesign, or stop.",
            "system_prompt": f"{common} Reject near-copy architectures and require written rationale for advancement.",
        },
    }
