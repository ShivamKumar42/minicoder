from __future__ import annotations
import os
from typing import Dict, Any, Optional
from .base import Tool


class WriteFileTool(Tool):
    def __init__(self, workspace: str):
        super().__init__()
        self.name = "write_file"
        self.workspace = os.path.abspath(workspace)

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within workspace to write/create"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }

    def execute(self, path: str, content: str) -> str:
        full_path = os.path.join(self.workspace, path)

        # Security: ensure file is within workspace
        abs_workspace = os.path.abspath(self.workspace)
        abs_full = os.path.abspath(full_path)

        if not abs_full.startswith(abs_workspace + os.sep) and abs_full != abs_workspace:
            return f"Error: Path escapes workspace: {path}"

        # Create parent directories if needed
        parent = os.path.dirname(full_path)
        os.makedirs(parent, exist_ok=True)

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"Error writing file: {str(e)}"

        return f"Successfully wrote to {path}"