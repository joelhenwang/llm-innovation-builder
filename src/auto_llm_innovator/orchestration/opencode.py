from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class OpenCodeAdapter:
    executable: str = "opencode"

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def command_preview(self, mode: str, prompt: str, cwd: Path) -> dict:
        if mode == "serve":
            command = [self.executable, "serve"]
        else:
            command = [self.executable, "run", prompt]
        return {"available": self.available(), "cwd": str(cwd), "command": command}

    def invoke(self, mode: str, prompt: str, cwd: Path) -> dict:
        preview = self.command_preview(mode=mode, prompt=prompt, cwd=cwd)
        if not preview["available"]:
            preview["status"] = "dry-run"
            preview["message"] = "OpenCode not installed; recorded orchestration preview only."
            return preview
        completed = subprocess.run(preview["command"], cwd=str(cwd), capture_output=True, text=True, check=False)
        preview["status"] = "completed" if completed.returncode == 0 else "failed"
        preview["returncode"] = completed.returncode
        preview["stdout"] = completed.stdout[-2000:]
        preview["stderr"] = completed.stderr[-2000:]
        return preview

    def to_dict(self) -> dict:
        return asdict(self)
