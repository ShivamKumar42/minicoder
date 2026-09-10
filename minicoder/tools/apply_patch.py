from __future__ import annotations
import os
from typing import Dict, Any
from .base import Tool
from .paths import is_within_workspace, resolve_workspace_path


class ApplyPatchTool(Tool):
    """Replace one exact `old_content` anchor with `patch` text.

    Semantics (matching how the agent actually calls this tool):
      - `filepath`: path relative to the workspace.
      - `old_content`: exact snippet currently in the file (anchor).
      - `patch`: replacement text for that snippet.

    Safety rules:
      - Empty `old_content` is rejected (it would match everywhere).
      - `old_content` must occur exactly once; zero matches or
        ambiguous multiple matches fail without touching the file.
      - Only the anchored snippet is replaced; the rest of the file
        is never overwritten.
    """

    def __init__(self, workspace: str):
        super().__init__()
        self.name = "apply_patch"
        self.description = "Replace an exact snippet in a file with new text"
        self.workspace = os.path.abspath(workspace)

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Relative path within workspace to modify"
                },
                "old_content": {
                    "type": "string",
                    "description": "Exact current snippet in the file to replace (must occur exactly once)"
                },
                "patch": {
                    "type": "string",
                    "description": "Replacement text for the old_content snippet"
                }
            },
            "required": ["filepath", "old_content", "patch"]
        }

    def execute(self, filepath: str, old_content: str, patch: str) -> str:
        full_path = resolve_workspace_path(self.workspace, filepath)

        # Security: ensure file is within workspace
        if not is_within_workspace(self.workspace, filepath):
            return f"Error: Path escapes workspace: {filepath}"

        if not os.path.isfile(full_path):
            return f"Error: File not found: {filepath}"

        if not old_content:
            return "Error: old_content must not be empty (no changes made)"

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                current_content = f.read()
        except Exception as e:
            return f"Error reading current file: {str(e)}"

        occurrences = current_content.count(old_content)
        if occurrences == 0:
            return f"Error: old_content not found in {filepath} (no changes made)"
        if occurrences > 1:
            return (
                f"Error: old_content matches {occurrences} locations in "
                f"{filepath}; must match exactly once (no changes made)"
            )

        new_content = current_content.replace(old_content, patch, 1)

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return f"Error writing patched file: {str(e)}"

        return f"Successfully applied patch to {filepath}"
