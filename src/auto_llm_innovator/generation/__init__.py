from __future__ import annotations

from pathlib import Path

from auto_llm_innovator.design_ir.models import DesignIR
from auto_llm_innovator.filesystem import ensure_dir, write_json, write_text
from auto_llm_innovator.generation.layout import build_idea_package_layout
from auto_llm_innovator.generation.postprocess import normalize_generated_source
from auto_llm_innovator.generation.renderers import render_idea_package_sources
from auto_llm_innovator.idea_spec.models import IdeaSpec


def generate_idea_package(idea_dir: Path, spec: IdeaSpec, design_ir: DesignIR) -> list[Path]:
    layout = build_idea_package_layout(idea_dir)
    for directory in (
        layout.package_dir,
        layout.package_modeling_dir,
        layout.package_evaluation_dir,
        layout.tests_dir,
    ):
        ensure_dir(directory)

    rendered = render_idea_package_sources(layout, spec, design_ir)
    written_paths: list[Path] = []
    for raw_path, content in rendered.items():
        path = Path(raw_path)
        write_text(path, normalize_generated_source(content))
        written_paths.append(path)

    manifest = {
        "generated_files": [str(path.relative_to(idea_dir)) for path in sorted(written_paths)],
        "design_ir_modules": [module.name for module in design_ir.modules],
        "evaluation_tasks": [task.name for task in design_ir.evaluation_plan],
    }
    write_json(layout.manifest_path, manifest)
    written_paths.append(layout.manifest_path)
    return written_paths


__all__ = ["generate_idea_package"]
