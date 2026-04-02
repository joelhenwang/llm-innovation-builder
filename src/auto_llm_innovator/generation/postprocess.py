from __future__ import annotations


def normalize_generated_source(content: str) -> str:
    lines = [line.rstrip() for line in content.splitlines()]
    normalized = "\n".join(lines).strip()
    return normalized + "\n"
