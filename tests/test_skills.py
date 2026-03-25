from pathlib import Path

from auto_llm_innovator.idea_spec import normalize_idea_spec
from auto_llm_innovator.skills import build_agent_prompt, doctor_skill_registry, explain_skill_profile, sync_reviewed_skills


def test_debugger_smoke_routing_includes_expected_skills():
    root = Path(__file__).resolve().parents[1]
    explanation = explain_skill_profile(root, "debugger", phase="smoke")
    active_names = {item["name"] for item in explanation["active"]}
    assert "systematic-debugging" in active_names
    assert "rocm-training-debugger" in active_names
    assert "smoke-test-math-and-shapes" in active_names
    assert "transformers" not in active_names


def test_reviewer_originality_guardrail_stays_strict():
    root = Path(__file__).resolve().parents[1]
    explanation = explain_skill_profile(root, "reviewer", phase="small")
    active_names = {item["name"] for item in explanation["active"]}
    assert "architecture-originality-gate" in active_names
    assert "transformers" not in explanation["forbidden"] or "transformers" not in active_names
    originality_skill = next(item for item in explanation["active"] if item["name"] == "architecture-originality-gate")
    assert any("generic transformer drift" in guardrail.lower() for guardrail in originality_skill["guardrails"])


def test_skill_registry_doctor_passes_for_repo_registry():
    root = Path(__file__).resolve().parents[1]
    doctor = doctor_skill_registry(root)
    assert doctor["valid"]
    assert doctor["counts"]["internal"] >= 11


def test_sync_only_includes_reviewed_external_skills(tmp_path: Path):
    sync = sync_reviewed_skills(tmp_path)
    assert sync["policy_enforced"]
    assert sync["ad_hoc_install_blocked"]
    assert all("url" in item for item in sync["installable_external_skills"])


def test_planner_prompt_skips_find_skills_by_default():
    root = Path(__file__).resolve().parents[1]
    spec = normalize_idea_spec("idea-0099", "Invent a new architecture with unusual routing.")
    prompt = build_agent_prompt(spec, role="planner", phase="smoke", root=root)
    active_names = {item["name"] for item in prompt.active_skills}
    injected_names = {item["name"] for item in prompt.injected_skills}
    skipped_names = {item["name"] for item in prompt.skipped_skills}
    assert "find-skills" in active_names
    assert "find-skills" not in injected_names
    assert "find-skills" in skipped_names


def test_implementer_prompt_injects_originality_and_skips_transformers_by_default():
    root = Path(__file__).resolve().parents[1]
    spec = normalize_idea_spec("idea-0100", "Invent a token mixer with originality safeguards.")
    prompt = build_agent_prompt(spec, role="implementer", phase="small", root=root)
    injected_names = {item["name"] for item in prompt.injected_skills}
    assert "architecture-originality-gate" in injected_names
    assert "deep-learning-python" in injected_names
    assert "transformers" not in injected_names


def test_implementer_prompt_can_enable_transformers_for_compatibility():
    root = Path(__file__).resolve().parents[1]
    spec = normalize_idea_spec("idea-0101", "Invent a tokenizer-aware recurrent architecture.")
    prompt = build_agent_prompt(
        spec,
        role="implementer",
        phase="small",
        root=root,
        context={"needs_tokenizer_api_compatibility": True},
    )
    injected_names = {item["name"] for item in prompt.injected_skills}
    assert "transformers" in injected_names


def test_debugger_smoke_prompt_injects_debug_skills():
    root = Path(__file__).resolve().parents[1]
    spec = normalize_idea_spec("idea-0102", "Invent a challenging architecture for smoke testing.")
    prompt = build_agent_prompt(spec, role="debugger", phase="smoke", root=root)
    injected_names = {item["name"] for item in prompt.injected_skills}
    assert "systematic-debugging" in injected_names
    assert "rocm-training-debugger" in injected_names
    assert "smoke-test-math-and-shapes" in injected_names
