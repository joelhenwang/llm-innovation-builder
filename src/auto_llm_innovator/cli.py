from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_llm_innovator.orchestration import InnovatorEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="innovator", description="Autonomous LLM innovator + trainer scaffold")
    parser.add_argument("--root", default=str(Path.cwd()), help="Project root containing ideas/ and baselines/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Submit a free-text idea")
    submit.add_argument("brief", help="Free-text idea brief")

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = InnovatorEngine(root=Path(args.root))

    if args.command == "submit":
        payload = engine.submit(args.brief).to_dict()
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

    parser.error(f"Unhandled command: {args.command}")
    return 2
