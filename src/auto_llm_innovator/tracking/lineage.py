from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArtifactHash:
    path: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DirectoryHash:
    path: str
    sha256: str
    bytes: int
    file_count: int
    files: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_file(path: Path) -> ArtifactHash:
    data = path.read_bytes()
    return ArtifactHash(
        path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )


def hash_directory(root: Path, *, relative_to: Path) -> DirectoryHash:
    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        artifact = hash_file(path)
        relative_path = _relative_path(path, relative_to)
        files[relative_path] = {
            "sha256": artifact.sha256,
            "bytes": artifact.bytes,
        }
        total_bytes += artifact.bytes
    return DirectoryHash(
        path=_relative_path(root, relative_to),
        sha256=hash_json_payload(files),
        bytes=total_bytes,
        file_count=len(files),
        files=files,
    )


def collect_artifact_record(path: Path, *, kind: str, relative_to: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": _relative_path(path, relative_to),
            "kind": kind,
            "missing": True,
        }
    if path.is_dir():
        directory_hash = hash_directory(path, relative_to=relative_to)
        return {
            "path": directory_hash.path,
            "kind": kind,
            "sha256": directory_hash.sha256,
            "bytes": directory_hash.bytes,
            "file_count": directory_hash.file_count,
            "files": directory_hash.files,
        }
    artifact_hash = hash_file(path)
    return {
        "path": _relative_path(path, relative_to),
        "kind": kind,
        "sha256": artifact_hash.sha256,
        "bytes": artifact_hash.bytes,
    }


def _relative_path(path: Path, relative_to: Path) -> str:
    try:
        return str(path.relative_to(relative_to))
    except ValueError:
        return str(path)
