from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_llm_innovator.handoff import HandoffValidationError
from auto_llm_innovator.orchestration import InnovatorEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="innovator", description="Autonomous LLM innovator + trainer scaffold")
    parser.add_argument("--root", default=str(Path.cwd()), help="Project root containing ideas/ and baselines/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Submit a free-text idea or structured bundle")
    submit_group = submit.add_mutually_exclusive_group(required=True)
    submit_group.add_argument("brief", nargs="?", help="Free-text idea brief")
    submit_group.add_argument("--bundle-file", help="Path to a structured researcher bundle JSON file")

    run = subparsers.add_parser("run", help="Run an idea lifecycle phase")
    run.add_argument("idea_id")
    run.add_argument("--phase", choices=["smoke", "small", "full", "all"], default="all")

    resume = subparsers.add_parser("resume", help="Resume the latest incomplete attempt")
    resume.add_argument("idea_id")

    status = subparsers.add_parser("status", help="Show idea status")
    status.add_argument("idea_id")

    compare = subparsers.add_parser("compare", help="Compare latest attempt against a baseline")
    compare.add_argument("idea_id")
    compare.add_argument("--baseline", default=None)

    report = subparsers.add_parser("report", help="Show latest decision report")
    report.add_argument("idea_id")

    skills = subparsers.add_parser("skills", help="Inspect curated skill registry and routing")
    skill_parsers = skills.add_subparsers(dest="skills_command", required=True)

    skill_parsers.add_parser("list", help="List reviewed skills in the registry")
    skill_parsers.add_parser("doctor", help="Validate the local skill registry and internal skill files")
    explain = skill_parsers.add_parser("explain", help="Explain which skills a role uses")
    explain.add_argument("role", choices=["planner", "implementer", "debugger", "trainer", "evaluator", "reviewer"])
    explain.add_argument("--phase", choices=["smoke", "small", "full"], default=None)
    explain.add_argument("--prompt-view", action="store_true", help="Show the prompt-builder payload instead of raw routing metadata")
    explain.add_argument("--idea-id", default=None, help="Optional idea id to use for prompt previews")
    skill_parsers.add_parser("sync", help="Materialize the reviewed external skill sync manifest")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = InnovatorEngine(root=Path(args.root))

    try:
        if args.command == "submit":
            payload = engine.submit(args.brief, bundle_file=args.bundle_file).to_dict()
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "run":
            print(json.dumps(engine.run(args.idea_id, phase=args.phase), indent=2))
            return 0
        if args.command == "resume":
            print(json.dumps(engine.resume(args.idea_id), indent=2))
            return 0
        if args.command == "status":
            print(json.dumps(engine.status(args.idea_id), indent=2))
            return 0
        if args.command == "compare":
            print(json.dumps(engine.compare(args.idea_id, baseline=args.baseline), indent=2))
            return 0
        if args.command == "report":
            print(engine.report(args.idea_id))
            return 0
        if args.command == "skills":
            if args.skills_command == "list":
                print(json.dumps(engine.skills_list(), indent=2))
                return 0
            if args.skills_command == "doctor":
                print(json.dumps(engine.skills_doctor(), indent=2))
                return 0
            if args.skills_command == "explain":
                if args.prompt_view:
                    phase = args.phase or "smoke"
                    print(json.dumps(engine.skills_prompt_view(args.role, phase=phase, idea_id=args.idea_id), indent=2))
                    return 0
                print(json.dumps(engine.skills_explain(args.role, phase=args.phase), indent=2))
                return 0
            if args.skills_command == "sync":
                print(json.dumps(engine.skills_sync(), indent=2))
                return 0
    except HandoffValidationError as exc:
        parser.exit(2, f"Validation error: {exc}\n")

    parser.error(f"Unhandled command: {args.command}")
    return 2
