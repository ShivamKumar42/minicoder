from __future__ import annotations
import os
import subprocess
import time
import re
import signal
from typing import Dict, Any, Optional, Tuple
from .base import Tool


class RunCommandTool(Tool):
    def __init__(self, workspace: str, timeout: int = 60,
                 blocked_commands: Optional[Set[str]] = None):
        super().__init__()
        self.name = "run_command"
        self.workspace = os.path.abspath(workspace)
        self.timeout = timeout
        self.blocked_commands = blocked_commands or {
            "sudo", "shutdown", "reboot", "mkfs", "dd",
        }

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
        """Check if command contains blocked patterns."""
        cmd_lower = command.lower().strip()
        for blocked in self.blocked_commands:
            if blocked in cmd_lower:
                return True
        # Check for fork bomb patterns
        if re.search(r':\s*\(\s*\$\)\(\s*\$\)', cmd_lower):
            return True
        # Check for dangerous piped commands
        if "|" in cmd_lower and ("rm" in cmd_lower or "del" in cmd_lower):
            return True
        return False

    def execute(self, command: str, cwd: Optional[str] = None,
                timeout: Optional[int] = None) -> Dict[str, Any]:
        full_cwd = self.workspace
        if cwd:
            cwd_path = os.path.join(self.workspace, cwd)
            if os.path.isdir(cwd_path):
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

        start_time = time.time()
        final_cwd = full_cwd

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=final_cwd,
                timeout=(timeout or self.timeout)
            )
            execution_time = time.time() - start_time
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
                "success": proc.returncode == 0,
                "execution_time": execution_time
            }
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                "stdout": "",
                "stderr": f"Error: Command timed out after {self.timeout}s",
                "exit_code": 124,
                "success": False,
                "execution_time": execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "stdout": "",
                "stderr": f"Error: {str(e)}",
                "exit_code": 1,
                "success": False,
                "execution_time": execution_time
            }