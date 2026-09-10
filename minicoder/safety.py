from __future__ import annotations
import os
import re
from typing import Any, Dict, FrozenSet, Iterable, Optional, Set

from .tools.base import Tool


# Shared block-lists. Individual callers keep their own membership
# (shell tool vs. approval mode) but share the matching logic below.
SHELL_BLOCKED_COMMANDS: FrozenSet[str] = frozenset(
    {"sudo", "shutdown", "reboot", "mkfs", "dd"}
)
APPROVAL_DANGEROUS_KEYWORDS: FrozenSet[str] = frozenset(
    {"rm -rf", "sudo", "shutdown", "format", "dd"}
)


def _normalize_for_matching(command: str) -> str:
    """Undo shell quoting tricks before block-list matching.

    The shell strips backslash escapes (su\\do -> sudo), empty quotes
    (s""udo -> sudo), and line continuations (su\\\ndo -> sudo) before
    executing, so matching must see the same text. Matching-only helper;
    the original command is never modified or executed.
    """
    text = re.sub(r"\\(.?)", r"\1", command, flags=re.DOTALL)
    text = text.replace("''", "").replace('""', "")
    return text


def _contains_token(command_lower: str, token: str) -> bool:
    """Match whole words for plain tokens, substrings otherwise.

    Plain alphanumeric tokens (e.g. "dd", "sudo") match on word
    boundaries so "address" is not flagged for "dd". Entries containing
    non-word characters (e.g. "rm -rf") keep substring semantics.
    """
    if re.fullmatch(r"[A-Za-z0-9_]+", token):
        return re.search(r"\b" + re.escape(token) + r"\b", command_lower) is not None
    return token in command_lower


def contains_blocked_command(
    command: str, blocked: Iterable[str] = SHELL_BLOCKED_COMMANDS
) -> bool:
    """Block-list match used by the shell tool (case-insensitive)."""
    cmd_lower = _normalize_for_matching(command).lower().strip()
    return any(_contains_token(cmd_lower, b) for b in blocked)


def is_fork_bomb(command: str) -> bool:
    """Detect fork-bomb patterns."""
    normalized = _normalize_for_matching(command).lower().strip()
    return re.search(r':\s*\(\s*\$\)\(\s*\$\)', normalized) is not None


def is_dangerous_pipe(command: str) -> bool:
    """Detect piped destructive commands (rm/del through a pipe)."""
    cmd_lower = _normalize_for_matching(command).lower().strip()
    return "|" in cmd_lower and ("rm" in cmd_lower or "del" in cmd_lower)


def is_blocked_command(
    command: str, blocked: Optional[Set[str]] = None
) -> bool:
    """Shell-tool policy: block-listed substrings, fork bombs, pipe-rm."""
    if contains_blocked_command(command, blocked or SHELL_BLOCKED_COMMANDS):
        return True
    if is_fork_bomb(command):
        return True
    if is_dangerous_pipe(command):
        return True
    return False


def contains_dangerous_keyword(
    command: str, keywords: Iterable[str] = APPROVAL_DANGEROUS_KEYWORDS
) -> bool:
    """Block-list match used by the agent approval mode (case-insensitive)."""
    cmd_lower = _normalize_for_matching(command).lower()
    return any(_contains_token(cmd_lower, kw) for kw in keywords)


class SafetyTool(Tool):
    """Safety check tool for path and command validation."""

    name = "safety_check"
    description = "Check whether a path or command is safe to use"

    def __init__(self, workspace: str):
        super().__init__()
        self.name = "safety_check"
        self.description = "Check whether a path or command is safe to use"
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