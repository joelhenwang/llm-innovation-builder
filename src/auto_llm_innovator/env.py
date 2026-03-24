from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(slots=True)
class EnvironmentReport:
    torch_available: bool
    rocm_available: bool
    device_count: int
    default_dtype: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def probe_environment() -> EnvironmentReport:
    try:
        import torch  # type: ignore
    except Exception:
        return EnvironmentReport(
            torch_available=False,
            rocm_available=False,
            device_count=0,
            default_dtype="float32",
            message="PyTorch not installed; framework will run in dry-run compatible mode.",
        )

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    cuda_available = bool(torch.cuda.is_available())
    rocm_available = bool(hip_version) and cuda_available
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    default_dtype = str(torch.get_default_dtype()).replace("torch.", "")
    message = "ROCm device available." if rocm_available else "PyTorch available; ROCm device not detected."
    return EnvironmentReport(
        torch_available=True,
        rocm_available=rocm_available,
        device_count=device_count,
        default_dtype=default_dtype,
        message=message,
    )
