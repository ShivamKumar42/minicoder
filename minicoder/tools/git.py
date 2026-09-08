from __future__ import annotations
import subprocess
import os
from typing import Dict, Any, Optional
from .base import Tool


class GitDiffTool(Tool):
    def __init__(self, workspace: str):
        super().__init__()
        self.name = "git_diff"
        self.workspace = os.path.abspath(workspace)

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
        full_path = self.workspace
        if path:
            target = os.path.join(self.workspace, path)
            if os.path.isdir(target):
                full_path = target

        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=full_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout if result.stdout else "No diff available"
            else:
                return f"Error running git diff: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: git diff timed out"
        except Exception as e:
            return f"Error running git diff: {str(e)}"


class GitStatusTool(Tool):
    def __init__(self, workspace: str):
        super().__init__()
        self.name = "git_status"
        self.workspace = os.path.abspath(workspace)

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
                return result.stdout if result.stdout else "No git status available"
            else:
                return f"Error running git status: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: git status timed out"
        except Exception as e:
            return f"Error running git status: {str(e)}"