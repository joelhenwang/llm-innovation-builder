from __future__ import annotations

import json
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

    def invoke_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_format_instructions: str,
        cwd: Path,
    ) -> dict:
        prompt = "\n".join(
            [
                system_prompt.rstrip(),
                "",
                response_format_instructions.strip(),
                "",
                user_prompt.rstrip(),
            ]
        ).strip()
        preview = self.command_preview(mode="run", prompt=prompt, cwd=cwd)
        if not preview["available"]:
            return {
                "role": role,
                "status": "dry-run",
                "command_preview": preview,
                "raw_stdout": "",
                "raw_stderr": "",
                "parsed_payload": None,
                "parse_status": "dry_run",
                "validation_errors": [],
            }
        completed = subprocess.run(preview["command"], cwd=str(cwd), capture_output=True, text=True, check=False)
        stdout = completed.stdout[-8000:]
        stderr = completed.stderr[-4000:]
        if completed.returncode != 0:
            return {
                "role": role,
                "status": "failed",
                "command_preview": preview,
                "raw_stdout": stdout,
                "raw_stderr": stderr,
                "parsed_payload": None,
                "parse_status": "runtime_failed",
                "validation_errors": [f"OpenCode returned non-zero exit status {completed.returncode}."],
            }
        parsed_payload, parse_errors = _parse_structured_stdout(stdout)
        return {
            "role": role,
            "status": "completed",
            "command_preview": preview,
            "raw_stdout": stdout,
            "raw_stderr": stderr,
            "parsed_payload": parsed_payload,
            "parse_status": "valid" if parsed_payload is not None else "invalid_json",
            "validation_errors": parse_errors,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_structured_stdout(stdout: str) -> tuple[dict | None, list[str]]:
    candidates = [stdout.strip()]
    fenced = _extract_fenced_json(stdout)
    if fenced:
        candidates.append(fenced)
    extracted = _extract_first_json_object(stdout)
    if extracted:
        candidates.append(extracted)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, []
    return None, ["OpenCode did not return a valid JSON object payload."]


def _extract_fenced_json(stdout: str) -> str | None:
    marker = "```json"
    start = stdout.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = stdout.find("```", start)
    if end == -1:
        return None
    return stdout[start:end].strip()


def _extract_first_json_object(stdout: str) -> str | None:
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(stdout):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return stdout[start : index + 1].strip()
    return None
