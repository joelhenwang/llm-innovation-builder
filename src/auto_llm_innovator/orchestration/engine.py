from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from auto_llm_innovator.constants import DEFAULT_MAX_AUTONOMOUS_RETRIES, PHASES, PROJECT_ROOT
from auto_llm_innovator.datasets import (
    apply_dataset_plan,
    dataset_plan_for_phase,
    persist_dataset_plan,
    plan_dataset_for_phase,
)
from auto_llm_innovator.design_ir import DesignIR, compile_design_ir, project_idea_spec, validate_design_ir
from auto_llm_innovator.env import EnvironmentReport, probe_environment
from auto_llm_innovator.evaluation import (
    BaselineDefinition,
    EvaluationResult,
    build_evaluation_result,
    compare_against_baseline,
    load_baseline_definition,
    render_decision_report,
)
from auto_llm_innovator.filesystem import ensure_dir, read_json, write_json, write_text
from auto_llm_innovator.generation import generate_idea_package
from auto_llm_innovator.handoff import load_research_idea_bundle
from auto_llm_innovator.idea_spec import IdeaSpec, normalize_idea_spec, review_originality
from auto_llm_innovator.modeling.interfaces import PhaseResult
from auto_llm_innovator.orchestration.agent_runtime import (
    AgentInvocationRecord,
    artifact_ref_for_path,
    build_agent_request_envelope,
    build_agent_response_artifact,
    persist_agent_request,
    persist_agent_response,
    persist_agent_runtime,
    validate_agent_payload,
)
from auto_llm_innovator.orchestration.agents import agent_definitions
from auto_llm_innovator.orchestration.opencode import OpenCodeAdapter
from auto_llm_innovator.planning import apply_phase_resource_plan, persist_resource_plan, plan_phase_resources
from auto_llm_innovator.runtime import default_runtime_settings_for_phase
from auto_llm_innovator.skills import (
    build_agent_prompt,
    doctor_skill_registry,
    explain_skill_profile,
    list_skills,
    persist_idea_skill_snapshot,
    sync_reviewed_skills,
)
from auto_llm_innovator.training import execute_phase
from auto_llm_innovator.tracking import build_phase_lineage_manifest, persist_phase_lineage_manifest
from auto_llm_innovator.tracking.ledger import create_attempt_record, finalize_attempt, load_status, record_phase_result
from auto_llm_innovator.tracking.ranking import AttemptRankingResult, build_attempt_ranking


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

    def submit(self, raw_brief: str | None = None, *, bundle_file: str | None = None) -> SubmitResult:
        idea_id = self._next_idea_id()
        idea_dir = ensure_dir(self.ideas_dir / idea_id)
        ensure_dir(idea_dir / "artifacts")
        ensure_dir(idea_dir / "runs")
        ensure_dir(idea_dir / "config")
        ensure_dir(idea_dir / "notes")
        ensure_dir(idea_dir / "reports")
        ensure_dir(idea_dir / "orchestration")

        if bundle_file is not None:
            bundle = load_research_idea_bundle(bundle_file=bundle_file)
        else:
            bundle = load_research_idea_bundle(raw_brief=raw_brief or "")
        design_ir = compile_design_ir(bundle, idea_id=idea_id)
        validate_design_ir(design_ir)
        spec = project_idea_spec(design_ir, bundle)
        originality = review_originality(spec)

        write_json(idea_dir / "handoff_bundle.json", bundle.to_dict())
        write_json(idea_dir / "design_ir.json", design_ir.to_dict())
        write_json(idea_dir / "idea_spec.json", spec.to_dict())
        write_json(idea_dir / "originality_review.json", originality.to_dict())
        write_json(idea_dir / "environment.json", probe_environment(self.root).to_dict())
        write_json(idea_dir / "opencode_preview.json", self.opencode.command_preview("run", spec.raw_brief, idea_dir))
        write_json(idea_dir / "orchestration" / "agents.json", agent_definitions(spec, root=self.root))
        persist_idea_skill_snapshot(self.root, idea_dir, spec.idea_id)
        write_text(idea_dir / "notes" / "design.md", self._design_notes(spec, originality))
        write_text(idea_dir / "notes" / "orchestration.md", self._orchestration_notes(spec))
        generate_idea_package(idea_dir, spec, design_ir)
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
            "## Design IR",
            "- See `design_ir.json` for the richer architecture, training, and evaluation planning artifact.",
            "",
            "## Originality review",
            f"- Passed: {originality.passed}",
            f"- Score: {originality.score}",
            *[f"- Revision: {item}" for item in originality.required_revisions],
        ]
        return "\n".join(lines) + "\n"

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
                "runtime": default_runtime_settings_for_phase("smoke").to_dict(),
                "novelty_claims": spec.novelty_claims,
                "max_retries_visible": DEFAULT_MAX_AUTONOMOUS_RETRIES,
            },
            "small": {
                "phase": "small",
                "target_parameters": 2_000_000_000,
                "prefer_rocm": True,
                "dataset": dataset_plan_for_phase("small"),
                "runtime": default_runtime_settings_for_phase("small").to_dict(),
                "novelty_claims": spec.novelty_claims,
                "max_retries_visible": DEFAULT_MAX_AUTONOMOUS_RETRIES,
            },
            "full": {
                "phase": "full",
                "target_parameters": spec.estimated_parameter_budget,
                "prefer_rocm": True,
                "dataset": dataset_plan_for_phase("full"),
                "runtime": default_runtime_settings_for_phase("full").to_dict(),
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
        previous_status = read_json(idea_dir / "status.json")
        create_attempt_record(idea_dir, attempt_id)
        phases = PHASES if phase == "all" else (phase,)
        results = []
        run_dir = ensure_dir(idea_dir / "runs" / attempt_id)
        design_ir = DesignIR.from_dict(read_json(idea_dir / "design_ir.json"))
        baseline_manifest = self.baselines_dir / "internal_reference" / "manifest.json"
        baseline_definition = load_baseline_definition(baseline_manifest)
        environment = self._load_environment_report(idea_dir)
        prior_ranking, prior_evaluation = self._latest_prior_context(idea_dir, previous_status)
        latest_prior_attempt_id = str(previous_status["attempts"][-1]["attempt_id"]) if previous_status.get("attempts") else None
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
            phase_config = read_json(idea_dir / "config" / f"{current_phase}.json")
            resource_plan = plan_phase_resources(
                design_ir=design_ir,
                phase_config=phase_config,
                environment=environment,
                baseline=baseline_definition,
                ranking_result=prior_ranking,
                evaluation_result=prior_evaluation,
            )
            resource_plan_path = persist_resource_plan(phase_dir, resource_plan)
            resolved_phase_config = apply_phase_resource_plan(phase_config, resource_plan)
            dataset_plan = plan_dataset_for_phase(
                design_ir=design_ir,
                phase=current_phase,
                phase_config=phase_config,
                resolved_phase_config=resolved_phase_config,
                resource_plan=resource_plan,
                baseline=baseline_definition,
            )
            dataset_plan_path = persist_dataset_plan(phase_dir, dataset_plan)
            resolved_phase_config = apply_dataset_plan(resolved_phase_config, dataset_plan)
            resolved_config_path = phase_dir / "resolved-config.json"
            write_json(resolved_config_path, resolved_phase_config)
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
            planner_request = build_agent_request_envelope(
                idea_id=idea_id,
                attempt_id=attempt_id,
                phase=current_phase,
                role="planner",
                expected_response_kind="planner_response",
                prompt_payload=prompt_payload["roles"]["planner"],
                context_artifacts=self._planner_context_artifacts(
                    idea_dir=idea_dir,
                    run_dir=run_dir,
                    previous_status=previous_status,
                    completed_results=results,
                    phase=current_phase,
                    phase_config_path=idea_dir / "config" / f"{current_phase}.json",
                    resource_plan_path=resource_plan_path,
                    dataset_plan_path=dataset_plan_path,
                    resolved_config_path=resolved_config_path,
                ),
                context={
                    "resource_admission_status": resource_plan.admission_status,
                    "dataset_executable": dataset_plan.executable,
                },
            )
            planner_artifacts = self._invoke_structured_role(
                idea_dir=idea_dir,
                phase_dir=phase_dir,
                request=planner_request,
            )
            if resource_plan.admission_status == "reject":
                result = self._rejected_phase_result(
                    idea_id=idea_id,
                    attempt_id=attempt_id,
                    phase=current_phase,
                    phase_dir=phase_dir,
                    resource_plan_path=resource_plan_path,
                    dataset_plan_path=dataset_plan_path,
                    resolved_config_path=resolved_config_path,
                    resource_plan=resource_plan,
                    dataset_plan=dataset_plan,
                )
            else:
                result = execute_phase(idea_dir=idea_dir, attempt_id=attempt_id, phase=current_phase, run_dir=phase_dir)
                result.artifacts_produced.append(str(resource_plan_path))
                result.artifacts_produced.append(str(dataset_plan_path))
                result.artifacts_produced.append(str(resolved_config_path))
                if resource_plan.warnings:
                    result.reviewer_notes.extend([f"Resource warning: {warning}" for warning in resource_plan.warnings])
                if resource_plan.admission_status == "downscale":
                    result.reviewer_notes.append(resource_plan.planner_summary)
                if dataset_plan.warnings:
                    result.reviewer_notes.extend([f"Dataset warning: {warning}" for warning in dataset_plan.warnings])
                if dataset_plan.reasons:
                    result.reviewer_notes.append(
                        f"Dataset plan: {' '.join(dataset_plan.reasons)}"
                    )
            result.artifacts_produced.append(str(prompt_path))
            result.artifacts_produced.append(str(skill_usage_path))
            result.artifacts_produced.extend(planner_artifacts)
            lineage_status = "rejected_before_execution" if resource_plan.admission_status == "reject" else "executed"
            lineage_manifest = build_phase_lineage_manifest(
                idea_dir=idea_dir,
                run_dir=phase_dir,
                idea_id=idea_id,
                attempt_id=attempt_id,
                phase=current_phase,
                lineage_status=lineage_status,
                environment=environment,
                result=result,
                resource_plan_path=resource_plan_path,
                dataset_plan_path=dataset_plan_path,
                resolved_config_path=resolved_config_path,
            )
            lineage_manifest_path = persist_phase_lineage_manifest(phase_dir, lineage_manifest)
            result.artifacts_produced.append(str(lineage_manifest_path))
            reviewer_request = build_agent_request_envelope(
                idea_id=idea_id,
                attempt_id=attempt_id,
                phase=current_phase,
                role="reviewer",
                expected_response_kind="reviewer_response",
                prompt_payload=prompt_payload["roles"]["reviewer"],
                context_artifacts=self._reviewer_context_artifacts(
                    idea_dir=idea_dir,
                    prompt_path=prompt_path,
                    skill_usage_path=skill_usage_path,
                    lineage_manifest_path=lineage_manifest_path,
                    latest_prior_attempt_id=latest_prior_attempt_id,
                ),
                context={
                    "phase_result": result.to_dict(),
                    "resource_admission_status": resource_plan.admission_status,
                    "latest_prior_attempt_id": latest_prior_attempt_id,
                },
            )
            reviewer_artifacts = self._invoke_structured_role(
                idea_dir=idea_dir,
                phase_dir=phase_dir,
                request=reviewer_request,
            )
            result.artifacts_produced.extend(reviewer_artifacts)
            result.reviewer_notes.append("Skill activation snapshot recorded.")
            result.reviewer_notes.append("Prompt payload recorded.")
            result.reviewer_notes.append("Planner structured artifact recorded.")
            result.reviewer_notes.append("Reviewer structured artifact recorded.")
            record_phase_result(idea_dir, result)
            results.append(result)
            if result.status != "passed":
                break

        evaluation_result = build_evaluation_result(
            idea_dir=idea_dir,
            attempt_id=attempt_id,
            baseline_manifest=baseline_manifest,
            results=results,
        )
        evaluation_path = idea_dir / "reports" / f"{attempt_id}-evaluation.json"
        write_json(evaluation_path, evaluation_result.to_dict())
        ranking_result = build_attempt_ranking(
            idea_dir=idea_dir,
            baseline=baseline_definition,
            evaluation_result=evaluation_result,
            status=read_json(idea_dir / "status.json"),
        )
        ranking_path = idea_dir / "reports" / f"{attempt_id}-ranking.json"
        write_json(ranking_path, ranking_result.to_dict())
        comparisons = compare_against_baseline(baseline_manifest, results, evaluation_result=evaluation_result)
        comparisons["ranking"] = ranking_result.to_dict()
        comparisons["prior_attempts"] = {
            "summary": ranking_result.prior_attempt_comparison_summary,
            "attempt_ids": list(ranking_result.prior_attempts_considered),
        }
        report_path = idea_dir / "reports" / f"{attempt_id}.md"
        render_decision_report(
            report_path,
            idea_id=idea_id,
            attempt_id=attempt_id,
            comparisons=comparisons,
            results=results,
            evaluation_result=evaluation_result,
        )
        status = read_json(idea_dir / "status.json")
        status["latest_report"] = str(report_path)
        write_json(idea_dir / "status.json", status)
        if any(result.status == "failed" for result in results):
            final_state = "failed"
        elif results and results[-1].phase == "full":
            final_state = "completed"
        else:
            final_state = "partial"
        finalize_attempt(idea_dir, attempt_id, final_state)
        return {
            "idea_id": idea_id,
            "attempt_id": attempt_id,
            "phases": [result.to_dict() for result in results],
            "evaluation_path": str(evaluation_path),
            "ranking_path": str(ranking_path),
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
            results.append(PhaseResult(**phase_payload))
        evaluation_path = idea_dir / "reports" / f"{latest_attempt['attempt_id']}-evaluation.json"
        evaluation_result = None
        if evaluation_path.exists():
            evaluation_result = EvaluationResult.from_dict(read_json(evaluation_path))
        comparison = compare_against_baseline(baseline_manifest, results, evaluation_result=evaluation_result)
        ranking_path = idea_dir / "reports" / f"{latest_attempt['attempt_id']}-ranking.json"
        if ranking_path.exists():
            ranking = AttemptRankingResult.from_dict(read_json(ranking_path))
            comparison["ranking"] = ranking.to_dict()
            comparison["prior_attempts"] = {
                "summary": ranking.prior_attempt_comparison_summary,
                "attempt_ids": list(ranking.prior_attempts_considered),
            }
        return comparison

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

    def _invoke_structured_role(
        self,
        *,
        idea_dir: Path,
        phase_dir: Path,
        request,
    ) -> list[str]:
        request_path = persist_agent_request(phase_dir, request)
        artifact_lines = [
            f"- {artifact.kind}: {artifact.path}" + (f" (sha256={artifact.sha256})" if artifact.sha256 else "")
            for artifact in request.context_artifacts
        ]
        supplemental_lines = []
        if artifact_lines:
            supplemental_lines.extend(["Referenced artifacts:"])
            supplemental_lines.extend(artifact_lines)
        if request.context:
            supplemental_lines.extend(["", "Structured context:", str(request.context)])
        invocation = self.opencode.invoke_structured(
            role=request.role,
            system_prompt=str(request.prompt_payload.get("system_prompt", "")),
            user_prompt="\n".join([str(request.prompt_payload.get("user_prompt", "")).rstrip(), "", *supplemental_lines]).strip(),
            response_format_instructions=request.response_format_instructions,
            cwd=idea_dir,
        )
        parse_status = str(invocation.get("parse_status", "invalid_json"))
        validation_errors = [str(item) for item in invocation.get("validation_errors", [])]
        raw_payload = invocation.get("parsed_payload")
        normalized_payload = None
        if parse_status == "valid":
            parse_status, normalized_payload, validation_errors = validate_agent_payload(
                role=request.role,
                payload=raw_payload if isinstance(raw_payload, dict) else None,
                allowed_artifact_paths={artifact.path for artifact in request.context_artifacts if not artifact.missing},
            )
        response_payload = build_agent_response_artifact(
            request=request,
            parse_status=parse_status,
            validation_errors=validation_errors,
            normalized_payload=normalized_payload,
            raw_payload=raw_payload if isinstance(raw_payload, dict) else None,
        )
        response_path = persist_agent_response(phase_dir, request.role, response_payload)
        runtime_path = persist_agent_runtime(
            phase_dir,
            AgentInvocationRecord(
                role=request.role,
                phase=request.phase,
                status=str(invocation.get("status", "failed")),
                parse_status=parse_status,
                validation_errors=validation_errors,
                command_preview=dict(invocation.get("command_preview", {})),
                raw_stdout=str(invocation.get("raw_stdout", "")),
                raw_stderr=str(invocation.get("raw_stderr", "")),
                parsed_payload=raw_payload if isinstance(raw_payload, dict) else None,
            ),
        )
        return [str(request_path), str(response_path), str(runtime_path)]

    def _planner_context_artifacts(
        self,
        *,
        idea_dir: Path,
        run_dir: Path,
        previous_status: dict,
        completed_results: list[PhaseResult],
        phase: str,
        phase_config_path: Path,
        resource_plan_path: Path,
        dataset_plan_path: Path,
        resolved_config_path: Path,
    ) -> list:
        artifacts = [
            artifact_ref_for_path(idea_dir / "design_ir.json", kind="design_ir", relative_to=idea_dir),
            artifact_ref_for_path(phase_config_path, kind="phase_config", relative_to=idea_dir),
            artifact_ref_for_path(resource_plan_path, kind="resource_plan", relative_to=idea_dir),
            artifact_ref_for_path(dataset_plan_path, kind="dataset_plan", relative_to=idea_dir),
            artifact_ref_for_path(resolved_config_path, kind="resolved_config", relative_to=idea_dir),
        ]
        for prior_result in completed_results:
            lineage_path = run_dir / prior_result.phase / "lineage-manifest.json"
            if lineage_path.exists():
                artifacts.append(
                    artifact_ref_for_path(lineage_path, kind="prior_phase_lineage", relative_to=idea_dir)
                )
        attempts = list(previous_status.get("attempts", []))
        if attempts:
            prior_attempt_id = str(attempts[-1]["attempt_id"])
            prior_lineage_path = idea_dir / "runs" / prior_attempt_id / phase / "lineage-manifest.json"
            if prior_lineage_path.exists():
                artifacts.append(
                    artifact_ref_for_path(prior_lineage_path, kind="prior_attempt_phase_lineage", relative_to=idea_dir)
                )
        return artifacts

    def _reviewer_context_artifacts(
        self,
        *,
        idea_dir: Path,
        prompt_path: Path,
        skill_usage_path: Path,
        lineage_manifest_path: Path,
        latest_prior_attempt_id: str | None,
    ) -> list:
        artifacts = [
            artifact_ref_for_path(lineage_manifest_path, kind="current_phase_lineage", relative_to=idea_dir),
            artifact_ref_for_path(prompt_path, kind="prompt_payload", relative_to=idea_dir),
            artifact_ref_for_path(skill_usage_path, kind="skills_payload", relative_to=idea_dir),
        ]
        if latest_prior_attempt_id is not None:
            ranking_path = idea_dir / "reports" / f"{latest_prior_attempt_id}-ranking.json"
            evaluation_path = idea_dir / "reports" / f"{latest_prior_attempt_id}-evaluation.json"
            if ranking_path.exists():
                artifacts.append(
                    artifact_ref_for_path(ranking_path, kind="prior_attempt_ranking", relative_to=idea_dir)
                )
            if evaluation_path.exists():
                artifacts.append(
                    artifact_ref_for_path(evaluation_path, kind="prior_attempt_evaluation", relative_to=idea_dir)
                )
        return artifacts

    def _load_environment_report(self, idea_dir: Path) -> EnvironmentReport:
        environment_path = idea_dir / "environment.json"
        if environment_path.exists():
            return EnvironmentReport.from_dict(read_json(environment_path))
        report = probe_environment(self.root)
        write_json(environment_path, report.to_dict())
        return report

    def _latest_prior_context(
        self,
        idea_dir: Path,
        status: dict,
    ) -> tuple[AttemptRankingResult | None, EvaluationResult | None]:
        attempts = list(status.get("attempts", []))
        if not attempts:
            return None, None
        latest_prior_attempt_id = str(attempts[-1]["attempt_id"])
        ranking_path = idea_dir / "reports" / f"{latest_prior_attempt_id}-ranking.json"
        evaluation_path = idea_dir / "reports" / f"{latest_prior_attempt_id}-evaluation.json"
        ranking = AttemptRankingResult.from_dict(read_json(ranking_path)) if ranking_path.exists() else None
        evaluation = EvaluationResult.from_dict(read_json(evaluation_path)) if evaluation_path.exists() else None
        return ranking, evaluation

    def _rejected_phase_result(
        self,
        *,
        idea_id: str,
        attempt_id: str,
        phase: str,
        phase_dir: Path,
        resource_plan_path: Path,
        dataset_plan_path: Path,
        resolved_config_path: Path,
        resource_plan,
        dataset_plan,
    ) -> PhaseResult:
        return PhaseResult(
            idea_id=idea_id,
            attempt_id=attempt_id,
            phase=phase,
            status="failed",
            key_metrics={},
            failure_signals=[*resource_plan.reasons, *resource_plan.warnings, *dataset_plan.reasons, *dataset_plan.warnings],
            artifacts_produced=[str(resource_plan_path), str(dataset_plan_path), str(resolved_config_path)],
            reviewer_notes=[
                "Phase execution was blocked by resource admission control before preflight started.",
                resource_plan.planner_summary,
                *([f"Dataset plan: {' '.join(dataset_plan.reasons)}"] if dataset_plan.reasons else []),
                *[f"Resource warning: {warning}" for warning in resource_plan.warnings],
                *[f"Dataset warning: {warning}" for warning in dataset_plan.warnings],
            ],
            next_action_recommendation="adjust_resources_or_stop",
            consumed_budget={
                "requested_parameters": resource_plan.resolved_target_parameters,
                "steps": 0,
                "device": "admission-blocked",
                "resumed": False,
                "stop_reason": "resource_admission_failed",
            },
            repair_attempted=False,
            repair_count=0,
            repair_outcome="not_attempted",
            failure_classification=None,
        )
