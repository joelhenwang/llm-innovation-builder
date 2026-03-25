from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from auto_llm_innovator.constants import DEFAULT_MAX_AUTONOMOUS_RETRIES, PHASES, PROJECT_ROOT
from auto_llm_innovator.datasets import dataset_plan_for_phase
from auto_llm_innovator.env import probe_environment
from auto_llm_innovator.evaluation import compare_against_baseline, render_decision_report
from auto_llm_innovator.filesystem import ensure_dir, read_json, write_json, write_text
from auto_llm_innovator.idea_spec import IdeaSpec, normalize_idea_spec, review_originality
from auto_llm_innovator.modeling.template import render_eval_template, render_model_template, render_train_template
from auto_llm_innovator.orchestration.agents import agent_definitions
from auto_llm_innovator.orchestration.opencode import OpenCodeAdapter
from auto_llm_innovator.skills import (
    build_agent_prompt,
    doctor_skill_registry,
    explain_skill_profile,
    list_skills,
    persist_idea_skill_snapshot,
    sync_reviewed_skills,
)
from auto_llm_innovator.training import execute_phase
from auto_llm_innovator.tracking.ledger import create_attempt_record, finalize_attempt, load_status, record_phase_result


@dataclass(slots=True)
class SubmitResult:
    idea_id: str
    idea_dir: str
    originality_passed: bool
    normalized_brief: str

    def to_dict(self) -> dict:
        return asdict(self)


class InnovatorEngine:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROJECT_ROOT
        self.ideas_dir = ensure_dir(self.root / "ideas")
        self.baselines_dir = ensure_dir(self.root / "baselines")
        self.opencode = OpenCodeAdapter()

    def _next_idea_id(self) -> str:
        existing = sorted(path.name for path in self.ideas_dir.iterdir() if path.is_dir() and path.name.startswith("idea-"))
        if not existing:
            return "idea-0001"
        return f"idea-{int(existing[-1].split('-')[-1]) + 1:04d}"

    def _next_attempt_id(self, idea_dir: Path) -> str:
        status_path = idea_dir / "status.json"
        if not status_path.exists():
            return "attempt-0001"
        status = read_json(status_path)
        count = len(status.get("attempts", [])) + 1
        return f"attempt-{count:04d}"

    def submit(self, raw_brief: str) -> SubmitResult:
        idea_id = self._next_idea_id()
        idea_dir = ensure_dir(self.ideas_dir / idea_id)
        ensure_dir(idea_dir / "artifacts")
        ensure_dir(idea_dir / "runs")
        ensure_dir(idea_dir / "config")
        ensure_dir(idea_dir / "notes")
        ensure_dir(idea_dir / "reports")
        ensure_dir(idea_dir / "orchestration")

        spec = normalize_idea_spec(idea_id=idea_id, raw_brief=raw_brief)
        originality = review_originality(spec)

        write_json(idea_dir / "idea_spec.json", spec.to_dict())
        write_json(idea_dir / "originality_review.json", originality.to_dict())
        write_json(idea_dir / "environment.json", probe_environment().to_dict())
        write_json(idea_dir / "opencode_preview.json", self.opencode.command_preview("run", spec.raw_brief, idea_dir))
        write_json(idea_dir / "orchestration" / "agents.json", agent_definitions(spec, root=self.root))
        persist_idea_skill_snapshot(self.root, idea_dir, spec.idea_id)
        write_text(idea_dir / "notes" / "design.md", self._design_notes(spec, originality))
        write_text(idea_dir / "notes" / "orchestration.md", self._orchestration_notes(spec))
        self._write_plugin_bundle(idea_dir, spec)
        self._write_phase_configs(idea_dir, spec)
        write_json(idea_dir / "status.json", {"idea_id": idea_id, "attempts": [], "latest_report": None})

        return SubmitResult(
            idea_id=idea_id,
            idea_dir=str(idea_dir),
            originality_passed=originality.passed,
            normalized_brief=spec.normalized_brief,
        )

    def _design_notes(self, spec: IdeaSpec, originality) -> str:
        lines = [
            f"# Design Notes: {spec.idea_id}",
            "",
            "## Raw brief",
            spec.raw_brief,
            "",
            "## Hypothesis",
            spec.hypothesis,
            "",
            "## Novelty claims",
            *[f"- {claim}" for claim in spec.novelty_claims],
            "",
            "## Forbidden fallback patterns",
            *[f"- {pattern}" for pattern in spec.forbidden_fallback_patterns],
            "",
            "## Originality review",
            f"- Passed: {originality.passed}",
            f"- Score: {originality.score}",
            *[f"- Revision: {item}" for item in originality.required_revisions],
        ]
        return "\n".join(lines) + "\n"

    def _write_plugin_bundle(self, idea_dir: Path, spec: IdeaSpec) -> None:
        write_text(idea_dir / "model.py", render_model_template(spec))
        write_text(idea_dir / "train.py", render_train_template(spec))
        write_text(idea_dir / "eval.py", render_eval_template())

    def _orchestration_notes(self, spec: IdeaSpec) -> str:
        return "\n".join(
            [
                f"# Orchestration Notes: {spec.idea_id}",
                "",
                "- Runtime: OpenCode CLI-first with optional `serve` mode.",
                "- Agents: planner, implementer, debugger, trainer, evaluator, reviewer.",
                "- Retry policy: open-ended exploration with visible retry metadata and lineage per attempt.",
                "- Tokenizer: GPT-2 for all phases and idea variants.",
                "- Originality: reviewer must block direct SOTA copies and generic fallback patterns.",
                "",
            ]
        )

    def _write_phase_configs(self, idea_dir: Path, spec: IdeaSpec) -> None:
        phase_configs = {
            "smoke": {
                "phase": "smoke",
                "target_parameters": 600_000_000,
                "prefer_rocm": True,
                "dataset": dataset_plan_for_phase("smoke"),
                "novelty_claims": spec.novelty_claims,
                "max_retries_visible": DEFAULT_MAX_AUTONOMOUS_RETRIES,
            },
            "small": {
                "phase": "small",
                "target_parameters": 2_000_000_000,
                "prefer_rocm": True,
                "dataset": dataset_plan_for_phase("small"),
                "novelty_claims": spec.novelty_claims,
                "max_retries_visible": DEFAULT_MAX_AUTONOMOUS_RETRIES,
            },
            "full": {
                "phase": "full",
                "target_parameters": spec.estimated_parameter_budget,
                "prefer_rocm": True,
                "dataset": dataset_plan_for_phase("full"),
                "novelty_claims": spec.novelty_claims,
                "max_retries_visible": DEFAULT_MAX_AUTONOMOUS_RETRIES,
            },
        }
        for phase, payload in phase_configs.items():
            write_json(idea_dir / "config" / f"{phase}.json", payload)

    def run(self, idea_id: str, phase: str = "all") -> dict:
        idea_dir = self.ideas_dir / idea_id
        if not idea_dir.exists():
            raise FileNotFoundError(f"Idea {idea_id} does not exist.")

        spec = IdeaSpec.from_dict(read_json(idea_dir / "idea_spec.json"))
        originality = read_json(idea_dir / "originality_review.json")
        if not originality["passed"]:
            raise RuntimeError(f"Idea {idea_id} failed originality review: {originality['required_revisions']}")

        attempt_id = self._next_attempt_id(idea_dir)
        create_attempt_record(idea_dir, attempt_id)
        phases = PHASES if phase == "all" else (phase,)
        results = []
        run_dir = ensure_dir(idea_dir / "runs" / attempt_id)
        roles = list(agent_definitions(spec, root=self.root))
        planner_prompt = build_agent_prompt(spec, role="planner", phase=phases[0], root=self.root)
        write_json(
            run_dir / "opencode_plan.json",
            {
                "role": "planner",
                "phase": phases[0],
                "prompt_payload": planner_prompt.to_dict(),
                "command_preview": self.opencode.command_preview(
                    "run", planner_prompt.system_prompt + "\n" + planner_prompt.user_prompt, idea_dir
                ),
            },
        )
        for current_phase in phases:
            phase_dir = ensure_dir(run_dir / current_phase)
            result = execute_phase(idea_dir=idea_dir, attempt_id=attempt_id, phase=current_phase, run_dir=phase_dir)
            prompt_payload = {
                "idea_id": idea_id,
                "attempt_id": attempt_id,
                "phase": current_phase,
                "roles": {
                    role: build_agent_prompt(spec, role=role, phase=current_phase, root=self.root).to_dict()
                    for role in roles
                },
            }
            prompt_path = phase_dir / "prompt.json"
            write_json(prompt_path, prompt_payload)
            skill_usage = {
                "phase": current_phase,
                "roles": {
                    role: {
                        "active_skills": prompt_payload["roles"][role]["active_skills"],
                        "injected_skills": prompt_payload["roles"][role]["injected_skills"],
                        "skipped_skills": prompt_payload["roles"][role]["skipped_skills"],
                    }
                    for role in roles
                },
            }
            skill_usage_path = phase_dir / "skills.json"
            write_json(skill_usage_path, skill_usage)
            result.artifacts_produced.append(str(prompt_path))
            result.artifacts_produced.append(str(skill_usage_path))
            result.reviewer_notes.append("Skill activation snapshot recorded.")
            result.reviewer_notes.append("Prompt payload recorded.")
            record_phase_result(idea_dir, result)
            results.append(result)

        comparisons = compare_against_baseline(self.baselines_dir / "internal_reference" / "manifest.json", results)
        report_path = idea_dir / "reports" / f"{attempt_id}.md"
        render_decision_report(report_path, idea_id=idea_id, attempt_id=attempt_id, comparisons=comparisons, results=results)
        status = read_json(idea_dir / "status.json")
        status["latest_report"] = str(report_path)
        write_json(idea_dir / "status.json", status)
        final_state = "completed" if phases[-1] == "full" else "partial"
        finalize_attempt(idea_dir, attempt_id, final_state)
        return {
            "idea_id": idea_id,
            "attempt_id": attempt_id,
            "phases": [result.to_dict() for result in results],
            "report_path": str(report_path),
        }

    def resume(self, idea_id: str) -> dict:
        idea_dir = self.ideas_dir / idea_id
        status = load_status(idea_dir)
        if not status["attempts"]:
            raise RuntimeError(f"Idea {idea_id} has no attempts to resume.")
        latest = status["attempts"][-1]
        for phase in PHASES:
            if phase not in latest["phases"]:
                return self.run(idea_id, phase=phase)
        return {"message": "Nothing to resume.", "attempt_id": latest["attempt_id"]}

    def status(self, idea_id: str) -> dict:
        idea_dir = self.ideas_dir / idea_id
        return load_status(idea_dir)

    def report(self, idea_id: str) -> str:
        idea_dir = self.ideas_dir / idea_id
        status = load_status(idea_dir)
        latest = status.get("latest_report")
        if not latest:
            raise RuntimeError(f"Idea {idea_id} has no report yet.")
        return Path(latest).read_text(encoding="utf-8")

    def compare(self, idea_id: str, baseline: str | None = None) -> dict:
        idea_dir = self.ideas_dir / idea_id
        status = load_status(idea_dir)
        if not status["attempts"]:
            raise RuntimeError(f"Idea {idea_id} has no runs to compare.")
        latest_attempt = status["attempts"][-1]
        baseline_manifest = self.baselines_dir / (baseline or "internal_reference") / "manifest.json"
        results = []
        for phase_payload in latest_attempt["phases"].values():
            from auto_llm_innovator.modeling.interfaces import PhaseResult

            results.append(PhaseResult(**phase_payload))
        return compare_against_baseline(baseline_manifest, results)

    def skills_list(self) -> dict:
        return list_skills(self.root)

    def skills_doctor(self) -> dict:
        return doctor_skill_registry(self.root)

    def skills_explain(self, role: str, phase: str | None = None) -> dict:
        return explain_skill_profile(self.root, role, phase=phase)

    def skills_prompt_view(self, role: str, phase: str, idea_id: str | None = None) -> dict:
        if idea_id is not None:
            idea_dir = self.ideas_dir / idea_id
            spec = IdeaSpec.from_dict(read_json(idea_dir / "idea_spec.json"))
        else:
            spec = normalize_idea_spec("preview-idea", f"Preview prompt for role={role} phase={phase}.")
        return build_agent_prompt(spec, role=role, phase=phase, root=self.root).to_dict()

    def skills_sync(self) -> dict:
        return sync_reviewed_skills(self.root)
