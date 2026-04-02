from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class IdeaPackageLayout:
    idea_dir: Path
    package_dir: Path
    package_modeling_dir: Path
    package_evaluation_dir: Path
    tests_dir: Path
    train_entrypoint: Path
    eval_entrypoint: Path
    manifest_path: Path


def build_idea_package_layout(idea_dir: Path) -> IdeaPackageLayout:
    package_dir = idea_dir / "package"
    return IdeaPackageLayout(
        idea_dir=idea_dir,
        package_dir=package_dir,
        package_modeling_dir=package_dir / "modeling",
        package_evaluation_dir=package_dir / "evaluation",
        tests_dir=idea_dir / "tests",
        train_entrypoint=idea_dir / "train.py",
        eval_entrypoint=idea_dir / "eval.py",
        manifest_path=idea_dir / "generation_manifest.json",
    )
