from __future__ import annotations
import os


class SafetyTool:
    """Safety check tool for path and command validation."""
    
    def __init__(self, workspace: str):
        self.workspace = os.path.abspath(workspace)

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "check": {
                    "type": "string",
                    "description": "What to check (path, command, or pattern)"
                }
            },
            "required": ["check"]
        }

    def execute(self, check: str) -> str:
        """Safety check for path or command."""
        # Check for path traversal
        if "../" in check or check.startswith("/"):
            return f"Security: Path traversal or absolute path detected: {check}"

        # Check for obviously destructive patterns
        check_lower = check.lower()
        destructive = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", ":()$()"]
        for pattern in destructive:
            if pattern in check_lower:
                return f"Security: Destructive command pattern detected: {check}"

        # Validate path stays within workspace
        abs_workspace = os.path.abspath(self.workspace)
        if check.startswith(abs_workspace):
            return f"OK: Path within workspace: {check}"

        # Relative path check
        abs_path = os.path.abspath(os.path.join(self.workspace, check))
        if abs_path.startswith(abs_workspace + os.sep) or abs_path == abs_workspace:
            return f"OK: Relative path within workspace: {check}"

        return f"Security: Path escapes workspace: {check}"