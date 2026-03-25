from pathlib import Path

from auto_llm_innovator.skills import doctor_skill_registry, explain_skill_profile, sync_reviewed_skills


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
