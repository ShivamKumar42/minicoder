from __future__ import annotations
import os
from typing import Dict, Any, Optional
from .base import Tool


class ReadFileTool(Tool):
    def __init__(self, workspace: str):
        super().__init__()
        self.name = "read_file"
        self.workspace = os.path.abspath(workspace)

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
        full_path = os.path.join(self.workspace, path)

        # Security: ensure file is within workspace
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace) + os.sep):
            return f"Error: Path escapes workspace: {path}"

        if not os.path.isfile(full_path):
            return f"Error: Not a file: {path}"

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading file: {str(e)}"

        total_lines = len(lines)
        actual_start = max(1, start_line)
        actual_end = min(total_lines, end_line) if end_line else total_lines

        if actual_start > total_lines:
            return f"Error: Start line {start_line} exceeds file length ({total_lines})"

        selected = lines[actual_start - 1:actual_end]
        return "".join(selected)