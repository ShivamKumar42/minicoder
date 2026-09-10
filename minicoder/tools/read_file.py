from __future__ import annotations
import itertools
import os
from typing import Dict, Any, Optional
from .base import Tool
from .budget import (
    DEFAULT_MAX_FILE_CHARS,
    DEFAULT_MAX_READ_LINES,
    truncate_text,
)
from .paths import is_within_workspace, resolve_workspace_path


class ReadFileTool(Tool):
    def __init__(
        self,
        workspace: str,
        max_chars: int = DEFAULT_MAX_FILE_CHARS,
        max_lines: int = DEFAULT_MAX_READ_LINES,
    ):
        super().__init__()
        self.name = "read_file"
        self.description = "Read a file with optional line ranges"
        self.workspace = os.path.abspath(workspace)
        self.max_chars = max_chars
        self.max_lines = max_lines

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within workspace to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read (1-indexed, default: 1)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read (inclusive, default: last line)"
                }
            },
            "required": ["path"]
        }

    def execute(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
        full_path = resolve_workspace_path(self.workspace, path)

        # Security: ensure file is within workspace
        if not is_within_workspace(self.workspace, path, allow_root=False):
            return f"Error: Path escapes workspace: {path}"

        if not os.path.isfile(full_path):
            return f"Error: Not a file: {path}"

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                # Page through the file instead of materializing it, so
                # large files cannot exhaust memory before truncation.
                total_lines = sum(1 for _ in f)
        except Exception as e:
            return f"Error reading file: {str(e)}"

        actual_start = max(1, start_line)
        actual_end = min(total_lines, end_line) if end_line else total_lines
        if actual_start > total_lines:
            return f"Error: Start line {start_line} exceeds file length ({total_lines})"
        line_cap = min(actual_end, actual_start + self.max_lines - 1)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                selected = list(
                    itertools.islice(f, actual_start - 1, line_cap)
                )
        except Exception as e:
            return f"Error reading file: {str(e)}"

        content = "".join(selected)
        hint = (
            f"(lines {actual_start}-{line_cap} of {total_lines}; "
            f"narrow start_line/end_line to read the rest)"
        )
        content, _ = truncate_text(content, self.max_chars, hint=hint)
        return content