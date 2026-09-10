"""Central workspace path guard for file tools.

All file tools must confine access to the configured workspace. These
helpers implement the check once so tools share identical semantics:
a joined path is allowed only if it resolves inside the workspace
(the workspace root itself counts as inside).

Paths are resolved with os.path.realpath so symlinks pointing outside
the workspace are rejected instead of followed.
"""

from __future__ import annotations
import os


def resolve_workspace_path(workspace: str, user_path: str) -> str:
    """Join a user-supplied path onto the workspace and resolve it.

    Uses realpath so the returned path has no symlinks, `..`, or `.`
    components; tools open exactly the location that was checked.
    """
    return os.path.realpath(os.path.join(os.path.abspath(workspace), user_path))


def is_within_workspace(
    workspace: str, user_path: str, allow_root: bool = True
) -> bool:
    """Return True if the joined path stays inside the workspace.

    With allow_root=True the workspace root itself counts as inside
    (used by tools that may target it); read_file uses
    allow_root=False to preserve its stricter historical check.
    Symlinks escaping the workspace resolve outside and return False.
    """
    abs_workspace = os.path.realpath(workspace)
    abs_full = resolve_workspace_path(workspace, user_path)
    if abs_full == abs_workspace:
        return allow_root
    return abs_full.startswith(abs_workspace + os.sep)
