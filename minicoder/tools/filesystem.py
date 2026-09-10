from __future__ import annotations
import os
import fnmatch
from typing import Dict, Any, List, Optional, Set
from .base import Tool
from .budget import (
    DEFAULT_MAX_LIST_CHARS,
    DEFAULT_MAX_LIST_FILES,
    truncate_text,
)
from .paths import is_within_workspace, resolve_workspace_path


class ListFilesTool(Tool):
    def __init__(
        self,
        workspace: str,
        ignored_dirs: Optional[Set[str]] = None,
        max_files: int = DEFAULT_MAX_LIST_FILES,
        max_chars: int = DEFAULT_MAX_LIST_CHARS,
    ):
        super().__init__()
        self.name = "list_files"
        self.description = "List files in the workspace, optionally recursively"
        self.workspace = os.path.abspath(workspace)
        self.ignored_dirs = ignored_dirs or {
            ".git", "node_modules", "__pycache__", ".venv",
            "dist", "build"
        }
        self.max_files = max_files
        self.max_chars = max_chars

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
            "required": []
        }

    def execute(self, path: str = ".", recursive: bool = True) -> str:
        if not is_within_workspace(self.workspace, path):
            return f"Error: Path escapes workspace: {path}"
        target = resolve_workspace_path(self.workspace, path)
        if not os.path.exists(target):
            return f"Error: Path does not exist: {path}"

        result: List[str] = []
        if not recursive:
            try:
                entries = sorted(os.listdir(target))
            except NotADirectoryError:
                return f"Error: Not a directory: {path}"
            for entry in entries:
                if entry in (self.ignored_dirs or set()):
                    continue
                full_path = os.path.join(target, entry)
                if os.path.isfile(full_path):
                    result.append(os.path.relpath(full_path, self.workspace))
            return self._format_result(result)

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

        return self._format_result(sorted(result))

    def _format_result(self, result: List[str]) -> str:
        total = len(result)
        shown = result[: self.max_files]
        output = "\n".join(shown)
        if total > len(shown):
            output += (
                f"\n... [{total - len(shown)} more files omitted; "
                f"list a subdirectory to narrow]"
            )
        output, _ = truncate_text(
            output,
            self.max_chars,
            hint="(list a subdirectory to narrow)",
        )
        return output