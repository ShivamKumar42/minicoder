from __future__ import annotations
import os
from typing import Dict, Any, Optional
from .base import Tool
from .paths import is_within_workspace, resolve_workspace_path


class WriteFileTool(Tool):
    def __init__(self, workspace: str):
        super().__init__()
        self.name = "write_file"
        self.description = "Create or overwrite a file with the given content"
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
        full_path = resolve_workspace_path(self.workspace, path)

        # Security: ensure file is within workspace
        if not is_within_workspace(self.workspace, path):
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