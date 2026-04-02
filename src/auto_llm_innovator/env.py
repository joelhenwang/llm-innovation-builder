from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EnvironmentReport:
    torch_available: bool
    accelerator_backend: str
    rocm_available: bool
    device_count: int
    gpu_names: list[str]
    vram_bytes_per_device: list[int]
    cpu_count: int
    system_ram_bytes: int
    free_disk_bytes: int
    default_dtype: str
    torch_version: str | None
    platform_system: str
    platform_machine: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnvironmentReport":
        return cls(
            torch_available=bool(payload["torch_available"]),
            accelerator_backend=str(payload.get("accelerator_backend", "none")),
            rocm_available=bool(payload.get("rocm_available", False)),
            device_count=int(payload.get("device_count", 0)),
            gpu_names=[str(item) for item in payload.get("gpu_names", [])],
            vram_bytes_per_device=[int(item) for item in payload.get("vram_bytes_per_device", [])],
            cpu_count=int(payload.get("cpu_count", 0)),
            system_ram_bytes=int(payload.get("system_ram_bytes", 0)),
            free_disk_bytes=int(payload.get("free_disk_bytes", 0)),
            default_dtype=str(payload.get("default_dtype", "float32")),
            torch_version=str(payload["torch_version"]) if payload.get("torch_version") is not None else None,
            platform_system=str(payload.get("platform_system", platform.system())),
            platform_machine=str(payload.get("platform_machine", platform.machine())),
            message=str(payload.get("message", "")),
        )


def probe_environment(root: Path | None = None) -> EnvironmentReport:
    cpu_count = os.cpu_count() or 0
    system_ram_bytes = _system_ram_bytes()
    free_disk_bytes = _free_disk_bytes(root or Path.cwd())
    platform_system = platform.system()
    platform_machine = platform.machine()
    try:
        import torch  # type: ignore
    except Exception:
        return EnvironmentReport(
            torch_available=False,
            accelerator_backend="none",
            rocm_available=False,
            device_count=0,
            gpu_names=[],
            vram_bytes_per_device=[],
            cpu_count=cpu_count,
            system_ram_bytes=system_ram_bytes,
            free_disk_bytes=free_disk_bytes,
            default_dtype="float32",
            torch_version=None,
            platform_system=platform_system,
            platform_machine=platform_machine,
            message="PyTorch not installed; no accelerator probe available, framework will run in dry-run compatible mode.",
        )

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    cuda_available = bool(torch.cuda.is_available())
    rocm_available = bool(hip_version) and cuda_available
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_names = []
    vram_bytes_per_device = []
    for index in range(device_count):
        try:
            properties = torch.cuda.get_device_properties(index)
            gpu_names.append(str(properties.name))
            vram_bytes_per_device.append(int(properties.total_memory))
        except Exception:
            gpu_names.append(f"device-{index}")
            vram_bytes_per_device.append(0)
    default_dtype = str(torch.get_default_dtype()).replace("torch.", "")
    accelerator_backend = "rocm" if rocm_available else "cuda" if cuda_available else "cpu"
    if accelerator_backend in {"cuda", "rocm"}:
        message = f"{accelerator_backend.upper()} available with {device_count} GPU(s)."
    else:
        message = "PyTorch available on CPU only; no accelerator detected."
    return EnvironmentReport(
        torch_available=True,
        accelerator_backend=accelerator_backend,
        rocm_available=rocm_available,
        device_count=device_count,
        gpu_names=gpu_names,
        vram_bytes_per_device=vram_bytes_per_device,
        cpu_count=cpu_count,
        system_ram_bytes=system_ram_bytes,
        free_disk_bytes=free_disk_bytes,
        default_dtype=default_dtype,
        torch_version=str(getattr(torch, "__version__", None)) if getattr(torch, "__version__", None) else None,
        platform_system=platform_system,
        platform_machine=platform_machine,
        message=message,
    )


def _system_ram_bytes() -> int:
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and page_count > 0:
                return page_size * page_count
        except (ValueError, OSError):
            pass
    return 0


def _free_disk_bytes(root: Path) -> int:
    try:
        return int(shutil.disk_usage(root).free)
    except OSError:
        return 0
