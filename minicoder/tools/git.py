from __future__ import annotations
import subprocess
import os
from typing import Dict, Any, Optional
from .base import Tool
from .budget import DEFAULT_MAX_OUTPUT_CHARS, truncate_text


class GitDiffTool(Tool):
    def __init__(self, workspace: str, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS):
        super().__init__()
        self.name = "git_diff"
        self.description = "Show uncommitted git changes"
        self.workspace = os.path.abspath(workspace)
        self.max_chars = max_chars

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within workspace (default: all changes)"
                }
            },
            "required": []
        }

    def execute(self, path: str = "") -> str:
        args = ["git", "diff"]
        if path:
            target = os.path.join(self.workspace, path)
            if not os.path.exists(target):
                return f"Error: Path does not exist: {path}"
            args += ["--", target]

        try:
            result = subprocess.run(
                args,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                if not result.stdout:
                    return "No diff available"
                output, _ = truncate_text(
                    result.stdout,
                    self.max_chars,
                    hint="(narrow path to see the rest)",
                )
                return output
            else:
                return f"Error running git diff: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: git diff timed out"
        except Exception as e:
            return f"Error running git diff: {str(e)}"


class GitStatusTool(Tool):
    def __init__(self, workspace: str, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS):
        super().__init__()
        self.name = "git_status"
        self.description = "Show git working-tree status"
        self.workspace = os.path.abspath(workspace)
        self.max_chars = max_chars

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "porcelain": {
                    "type": "boolean",
                    "description": "Return porcelain format (default: false)"
                }
            },
            "required": []
        }

    def execute(self, porcelain: bool = False) -> str:
        try:
            if porcelain:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            else:
                result = subprocess.run(
                    ["git", "status"],
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

            if result.returncode == 0:
                if not result.stdout:
                    return "No git status available"
                output, _ = truncate_text(
                    result.stdout,
                    self.max_chars,
                    hint="(narrow the workspace to see the rest)",
                )
                return output
            else:
                return f"Error running git status: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: git status timed out"
        except Exception as e:
            return f"Error running git status: {str(e)}"