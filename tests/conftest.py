from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_llm_innovator.env import EnvironmentReport
from auto_llm_innovator.orchestration.opencode import OpenCodeAdapter


@pytest.fixture(autouse=True)
def disable_real_opencode_for_tests(monkeypatch, request):
    if request.module.__name__.endswith("test_opencode"):
        return
    monkeypatch.setattr(OpenCodeAdapter, "available", lambda self: False)


@pytest.fixture
def bootstrap_baseline():
    def _bootstrap(root: Path) -> Path:
        baseline_dir = root / "baselines" / "internal_reference"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = baseline_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "baseline_id": "internal-reference-v1",
                    "family": "internal_reference",
                    "label": "Internal Reference",
                    "metric_targets": [
                        {"phase": "smoke", "metric_name": "loss", "target_value": 6.0},
                        {"phase": "small", "metric_name": "val_loss", "target_value": 4.2},
                        {"phase": "full", "metric_name": "val_loss", "target_value": 3.7},
                    ],
                    "token_budget_assumptions": {
                        "smoke": 50_000,
                        "small": 3_000_000,
                        "full": 12_000_000,
                    },
                    "reference_metrics": {
                        "smoke.loss": 6.0,
                        "small.val_loss": 4.2,
                        "full.val_loss": 3.7,
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest_path

    return _bootstrap


@pytest.fixture
def write_environment():
    def _write(
        idea_dir: Path,
        *,
        backend: str = "cuda",
        vram_bytes_per_device: list[int] | None = None,
        system_ram_bytes: int = 64_000_000_000,
    ) -> EnvironmentReport:
        report = EnvironmentReport(
            torch_available=backend != "none",
            accelerator_backend=backend,
            rocm_available=backend == "rocm",
            device_count=len(vram_bytes_per_device or []),
            gpu_names=[f"{backend}-gpu-{index}" for index in range(len(vram_bytes_per_device or []))],
            vram_bytes_per_device=list(vram_bytes_per_device or []),
            cpu_count=8,
            system_ram_bytes=system_ram_bytes,
            free_disk_bytes=400_000_000_000,
            default_dtype="float32",
            torch_version="2.8.0" if backend != "none" else None,
            platform_system="Darwin",
            platform_machine="arm64",
            message="test environment",
        )
        (idea_dir / "environment.json").write_text(json.dumps(report.to_dict()), encoding="utf-8")
        return report

    return _write


@pytest.fixture
def structured_agent_stub(monkeypatch):
    def _install(
        *,
        planner: dict | None = None,
        reviewer: dict | None = None,
    ) -> None:
        def _default_payload(role: str) -> dict:
            if role == "planner":
                return {
                    "phase_summary": "Plan the phase execution.",
                    "focus_areas": ["runtime wiring"],
                    "risk_flags": ["budget drift"],
                    "success_criteria": ["phase completes"],
                    "recommended_next_action": "execute_phase",
                }
            return {
                "recommendation": "continue_with_caution",
                "summary": "The run is usable for follow-up work.",
                "blocking_concerns": [],
                "continuation_criteria": ["keep tracking lineage"],
                "cited_artifacts": [],
            }

        def _payload_for(role: str) -> dict:
            config = planner if role == "planner" else reviewer
            payload = _default_payload(role)
            if config:
                payload.update(config)
            return payload

        def _invoke(self, **kwargs):
            role = kwargs["role"]
            payload = _payload_for(role)
            command = ["opencode", "run", role]
            return {
                "status": str(payload.pop("status", "completed")),
                "command_preview": payload.pop(
                    "command_preview",
                    {"available": True, "cwd": str(kwargs["cwd"]), "command": command},
                ),
                "raw_stdout": str(payload.pop("raw_stdout", json.dumps(payload.get("parsed_payload", payload)))),
                "raw_stderr": str(payload.pop("raw_stderr", "")),
                "parsed_payload": payload.pop("parsed_payload", payload),
                "parse_status": str(payload.pop("parse_status", "valid")),
                "validation_errors": [str(item) for item in payload.pop("validation_errors", [])],
            }

        monkeypatch.setattr(OpenCodeAdapter, "invoke_structured", _invoke)

    return _install


@pytest.fixture
def assert_phase_agent_artifacts():
    def _assert(phase_dir: Path, phase_payload: dict | None = None) -> dict:
        loaded: dict[str, dict[str, dict]] = {}
        for role in ("planner", "reviewer"):
            request_path = phase_dir / "agents" / f"{role}-request.json"
            response_path = phase_dir / "agents" / f"{role}-response.json"
            runtime_path = phase_dir / "agents" / f"{role}-runtime.json"
            assert request_path.exists()
            assert response_path.exists()
            assert runtime_path.exists()
            if phase_payload is not None:
                artifacts = phase_payload["artifacts_produced"]
                assert str(request_path) in artifacts
                assert str(response_path) in artifacts
                assert str(runtime_path) in artifacts
            loaded[role] = {
                "request": json.loads(request_path.read_text(encoding="utf-8")),
                "response": json.loads(response_path.read_text(encoding="utf-8")),
                "runtime": json.loads(runtime_path.read_text(encoding="utf-8")),
            }
        return loaded

    return _assert
