from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeContext:
    torch_available: bool
    rocm_available: bool
    dry_run: bool


def detect_runtime() -> RuntimeContext:
    try:
        import torch  # type: ignore
    except Exception:
        return RuntimeContext(torch_available=False, rocm_available=False, dry_run=True)

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    rocm_available = bool(hip_version) and bool(torch.cuda.is_available())
    return RuntimeContext(torch_available=True, rocm_available=rocm_available, dry_run=not rocm_available)
