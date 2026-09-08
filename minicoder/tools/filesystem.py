from __future__ import annotations
import os
import fnmatch
from typing import Dict, Any, List, Optional, Set
from .base import Tool
from ..messages import Message


class ListFilesTool(Tool):
    def __init__(self, workspace: str, ignored_dirs: Optional[Set[str]] = None):
        super().__init__()
        self.name = "list_files"
        self.workspace = os.path.abspath(workspace)
        self.ignored_dirs = ignored_dirs or {
            ".git", "node_modules", "__pycache__", ".venv",
            "dist", "build"
        }

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within workspace to list (default: root)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to recursively list subdirectories"
                }
            },
            "required": ["path", "recursive"]
        }

    def execute(self, path: str = ".", recursive: bool = True) -> str:
        target = os.path.join(self.workspace, path)
        if not os.path.exists(target):
            return f"Error: Path does not exist: {path}"

        result: List[str] = []
        for root, dirs, files in os.walk(target):
            # Filter ignored directories
            dirs[:] = [d for d in dirs if not any(
                fnmatch.fnmatch(d, ig) for ig in self.ignored_dirs
            )]

            # Also filter by base name
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]

            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, self.workspace)
                result.append(rel_path)

        return "\n".join(sorted(result))