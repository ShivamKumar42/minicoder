from __future__ import annotations
import os
import subprocess
import time
from typing import Dict, Any, Optional, Set
from ..safety import SHELL_BLOCKED_COMMANDS, is_blocked_command
from .base import Tool
from .budget import DEFAULT_MAX_OUTPUT_CHARS, truncate_text
from .paths import is_within_workspace


class RunCommandTool(Tool):
    # Hard cap per stream so one command cannot flood the LLM context.
    # Kept as a class attribute for backwards compatibility; prefer the
    # `max_output_chars` constructor argument for configuration.
    MAX_OUTPUT_CHARS = DEFAULT_MAX_OUTPUT_CHARS

    def __init__(self, workspace: str, timeout: int = 60,
                 blocked_commands: Optional[Set[str]] = None,
                 max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS):
        super().__init__()
        self.name = "run_command"
        self.description = "Execute a shell command and return its output"
        self.workspace = os.path.abspath(workspace)
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.blocked_commands = blocked_commands or set(SHELL_BLOCKED_COMMANDS)

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (relative to workspace, optional)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Override default timeout in seconds"
                }
            },
            "required": ["command"]
        }

    def _is_blocked(self, command: str) -> bool:
        """Check if command contains blocked patterns (see safety.py)."""
        return is_blocked_command(command, self.blocked_commands)

    def _truncate(self, text: Optional[str]) -> tuple[str, bool]:
        """Cap a stream at max_output_chars, keeping the head."""
        return truncate_text(text, self.max_output_chars)

    def execute(self, command: str, cwd: Optional[str] = None,
                timeout: Optional[int] = None) -> Dict[str, Any]:
        full_cwd = self.workspace
        if cwd:
            cwd_path = os.path.join(self.workspace, cwd)
            # Confinement: ignore directories outside the workspace
            # (matches the fallback for nonexistent directories). Note
            # `cd` inside the command string itself is the caller's
            # responsibility and is not constrained here.
            if os.path.isdir(cwd_path) and is_within_workspace(
                self.workspace, cwd
            ):
                full_cwd = cwd_path

        # Check for blocked commands
        if self._is_blocked(command):
            return {
                "stdout": "",
                "stderr": f"Error: Command blocked for safety: {command[:50]}...",
                "exit_code": 1,
                "success": False,
                "execution_time": 0
            }

        effective_timeout = timeout or self.timeout
        start_time = time.time()
        final_cwd = full_cwd

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=final_cwd,
                timeout=effective_timeout
            )
            execution_time = time.time() - start_time
            stdout, stdout_truncated = self._truncate(proc.stdout)
            stderr, stderr_truncated = self._truncate(proc.stderr)
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": proc.returncode,
                "success": proc.returncode == 0,
                "execution_time": execution_time,
                "truncated": stdout_truncated or stderr_truncated,
            }
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            stdout, _ = self._truncate(e.stdout if isinstance(e.stdout, str) else "")
            stderr, _ = self._truncate(e.stderr if isinstance(e.stderr, str) else "")
            err = f"Error: Command timed out after {effective_timeout}s"
            if stderr:
                err = f"{err}\n{stderr}"
            return {
                "stdout": stdout,
                "stderr": err,
                "exit_code": 124,
                "success": False,
                "execution_time": execution_time,
                "truncated": True,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "stdout": "",
                "stderr": f"Error: {str(e)}",
                "exit_code": 1,
                "success": False,
                "execution_time": execution_time,
                "truncated": False,
            }