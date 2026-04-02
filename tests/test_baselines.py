import json
from pathlib import Path

from auto_llm_innovator.evaluation import load_baseline_definition


def test_load_legacy_manifest_normalizes_to_structured_definition(tmp_path: Path):
    baseline_dir = tmp_path / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "baseline_id": "internal-reference-v1",
                "description": "Legacy manifest",
                "tokenizer": "gpt2",
                "reference_metrics": {
                    "smoke.loss": 6.0,
                    "small.val_loss": 4.2,
                },
            }
        ),
        encoding="utf-8",
    )

    baseline = load_baseline_definition(path)

    assert baseline.baseline_id == "internal-reference-v1"
    assert baseline.family == "internal_reference"
    assert baseline.tokenizer == "gpt2"
    assert baseline.reference_metrics()["smoke.loss"] == 6.0
    assert baseline.reference_metrics()["small.val_loss"] == 4.2


def test_load_structured_manifest_preserves_optional_fields(tmp_path: Path):
    baseline_dir = tmp_path / "baselines" / "family_a"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "baseline_id": "family-a-v2",
                "family": "recurrent_memory",
                "label": "Family A",
                "tokenizer": "gpt2",
                "metric_targets": [
                    {"phase": "smoke", "metric_name": "loss", "target_value": 5.9},
                    {"phase": "small", "metric_name": "val_loss", "target_value": 4.0},
                ],
                "reliability_expectations": {"smoke": "clean_pass"},
                "practicality_expectations": {"small": "no_budget_limited"},
                "hardware_assumptions": {"device": "rocm"},
                "token_budget_assumptions": {"small": 1000000},
            }
        ),
        encoding="utf-8",
    )

    baseline = load_baseline_definition(path)

    assert baseline.family == "recurrent_memory"
    assert baseline.label == "Family A"
    assert baseline.reliability_expectations["smoke"] == "clean_pass"
    assert baseline.practicality_expectations["small"] == "no_budget_limited"
    assert baseline.hardware_assumptions["device"] == "rocm"
    assert baseline.token_budget_assumptions["small"] == 1000000


def test_load_structured_manifest_allows_missing_optional_fields(tmp_path: Path):
    baseline_dir = tmp_path / "baselines" / "minimal"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "baseline_id": "minimal-v1",
                "metric_targets": [{"phase": "full", "metric_name": "val_loss", "target_value": 3.7}],
            }
        ),
        encoding="utf-8",
    )

    baseline = load_baseline_definition(path)

    assert baseline.family == "internal_reference"
    assert baseline.reference_metrics()["full.val_loss"] == 3.7
    assert baseline.reliability_expectations == {}
    assert baseline.practicality_expectations == {}
