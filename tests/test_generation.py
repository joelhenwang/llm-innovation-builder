import importlib.util
import json
import sys
from pathlib import Path

from auto_llm_innovator.design_ir import compile_design_ir
from auto_llm_innovator.generation import generate_idea_package
from auto_llm_innovator.handoff import load_research_idea_bundle
from auto_llm_innovator.idea_spec import review_originality
from auto_llm_innovator.design_ir import project_idea_spec


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    import_root = path.parent.parent if path.parent.name == "package" else path.parent
    sys.path.insert(0, str(import_root))
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(import_root):
            sys.path.pop(0)
    return module


def _compile_submission_artifacts(raw_brief: str, idea_id: str = "idea-0001"):
    bundle = load_research_idea_bundle(raw_brief=raw_brief)
    design_ir = compile_design_ir(bundle, idea_id=idea_id)
    spec = project_idea_spec(design_ir, bundle)
    review_originality(spec)
    return spec, design_ir


def test_generation_emits_package_layout_and_manifest(tmp_path: Path):
    idea_dir = tmp_path / "ideas" / "idea-0001"
    spec, design_ir = _compile_submission_artifacts(
        "Invent a recurrent retrieval decoder with cache-aware routing and explicit memory."
    )

    written = generate_idea_package(idea_dir, spec, design_ir)

    assert (idea_dir / "package" / "__init__.py").exists()
    assert (idea_dir / "package" / "plugin.py").exists()
    assert (idea_dir / "package" / "modeling" / "components.py").exists()
    assert (idea_dir / "package" / "modeling" / "state.py").exists()
    assert (idea_dir / "package" / "evaluation" / "hooks.py").exists()
    assert (idea_dir / "tests" / "test_shapes.py").exists()
    assert (idea_dir / "train.py").exists()
    assert (idea_dir / "eval.py").exists()
    manifest = json.loads((idea_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert "package/plugin.py" in manifest["generated_files"]
    assert len(written) == len(manifest["generated_files"]) + 1


def test_generation_omits_state_file_when_design_ir_has_no_state_semantics(tmp_path: Path):
    idea_dir = tmp_path / "ideas" / "idea-0002"
    spec, design_ir = _compile_submission_artifacts("Invent a novel decoder-only mixer with gated residual token blending.")

    generate_idea_package(idea_dir, spec, design_ir)
    plugin_module = _load_module("idea_0002_plugin", idea_dir / "package" / "plugin.py")

    assert not (idea_dir / "package" / "modeling" / "state.py").exists()
    descriptor = plugin_module.describe_plugin()
    assert descriptor["supports"] == {
        "recurrent_state": False,
        "external_memory": False,
        "cache_path": False,
    }


def test_generation_plugin_exports_runtime_contract_and_evaluation_tasks(tmp_path: Path):
    idea_dir = tmp_path / "ideas" / "idea-0003"
    spec, design_ir = _compile_submission_artifacts(
        "Invent a recurrent retrieval decoder that compares against baseline and prior work with cache-aware routing."
    )

    generate_idea_package(idea_dir, spec, design_ir)
    plugin_module = _load_module("idea_0003_plugin", idea_dir / "package" / "plugin.py")
    train_contents = (idea_dir / "train.py").read_text(encoding="utf-8")

    assert hasattr(plugin_module, "ModelConfig")
    assert callable(plugin_module.build_model)
    assert callable(plugin_module.describe_plugin)
    assert callable(plugin_module.register_evaluation_hooks)
    descriptor = plugin_module.describe_plugin()
    assert descriptor["module_names"]
    hook_names = sorted(plugin_module.register_evaluation_hooks())
    assert hook_names == sorted(task.name for task in design_ir.evaluation_plan)
    assert "from package import plugin as plugin_module" in train_contents
