from __future__ import annotations
import os
import re
import posixpath
from typing import Dict, Any, Optional, List, Tuple
from .base import Tool


class ApplyPatchTool(Tool):
    def __init__(self, workspace: str):
        super().__init__()
        self.name = "apply_patch"
        self.workspace = os.path.abspath(workspace)

    def _nby_line_diff(self, old: str, new: str) -> Tuple[int, int]:
        """Find the line range that changed between old and new content."""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        # Find first differing line
        first_diff = 0
        for i, (o, n) in enumerate(zip(old_lines, new_lines)):
            if o != n:
                first_diff = i
                break

        if first_diff >= len(old_lines) and first_diff >= len(new_lines):
            return (0, 0)

        # Find last differing line
        last_diff = max(len(old_lines), len(new_lines)) - 1
        for i in range(min(len(old_lines), len(new_lines)) - 1, -1, -1):
            if old_lines[i] != new_lines[i]:
                last_diff = i
                break

        return (first_diff, last_diff)

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
                    "description": "Current content of the file (for patch matching)"
                },
                "patch": {
                    "type": "string",
                    "description": "Patch content to apply (unified diff format or hunk)"
                }
            },
            "required": ["filepath", "old_content", "patch"]
        }

    def execute(self, filepath: str, old_content: str, patch: str) -> str:
        full_path = os.path.join(self.workspace, filepath)

        # Security: ensure file is within workspace
        abs_workspace = os.path.abspath(self.workspace)
        abs_full = os.path.abspath(full_path)

        if not abs_full.startswith(abs_workspace + os.sep) and abs_full != abs_workspace:
            return f"Error: Path escapes workspace: {filepath}"

        if not os.path.isfile(full_path):
            return f"Error: File not found: {filepath}"

        try:
            current_content = self._read_file_safely(full_path)
        except Exception as e:
            return f"Error reading current file: {str(e)}"

        new_content = self._apply_patch_content(current_content, patch)

        if new_content is None:
            return f"Error: Patch could not be applied to {filepath}"

        # Write the new content
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return f"Error writing patched file: {str(e)}"

        return f"Successfully applied patch to {filepath}"

    def _read_file_safely(self, full_path: str) -> str:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def _apply_patch_content(self, content: str, patch: str) -> Optional[str]:
        """Apply a unified diff patch to file content."""
        lines = content.splitlines(keepends=True)
        patch_lines = patch.splitlines()

        # Try to parse unified diff format
        if patch_lines and patch_lines[0].startswith("---"):
            return self._apply_unified_diff(content, patch)

        # Try as a simple hunk
        return self._apply_simple_hunk(content, patch)

    def _apply_unified_diff(self, content: str, patch: str) -> Optional[str]:
        """Apply a unified diff patch."""
        lines = content.splitlines(keepends=True)
        diff_lines = patch.splitlines()

        # Remove --- and +++ headers
        filtered = []
        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                i += 1
                continue
            filtered.append(line)
            i += 1

        # Parse hunk headers and apply
        # This is a simplified parser - finds @@ markers and applies changes
        # For robust patching, we use line-by-line comparison

        # Try to match the old file content
        if not filtered:
            return None

        # Check if the first few lines match
        old_start = 0
        for i, pl in enumerate(filtered):
            if pl.startswith("@@"):
                old_start = i
                break

        if old_start == 0:
            return None

        # Extract the old and new sections from the hunk
        # This is a simplified approach - just replace the matching section
        result_lines = list(lines)

        # Parse the hunk to find line numbers and changes
        current_line = old_start
        new_lines_list: List[str] = []
        removed_lines: List[int] = []

        for fl in filtered:
            if fl.startswith("@@"):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.match(r"@@ -\d+(?:,\d*)? \d+(?:,\d*)? @@", fl)
                if match:
                    # Simple extraction - get the changes after this line
                    remaining = filtered[filtered.index(fl) + 1:]
                    # Apply changes line by line
                    new_content = self._apply_hunk_lines(lines, remaining)
                    if new_content is not None:
                        return new_content
                continue
            # Skip other non-content lines like "Files changed" etc.

        return None

    def _apply_hunk_lines(self, original: List[str], hunk_lines: List[str]) -> Optional[str]:
        """Apply a hunk of changes to original lines."""
        result = list(original)
        orig_idx = 0

        for hln in hunk_lines:
            hln = hln.strip()
            if not hln:
                continue
            if hln.startswith("+"):
                # Insert new line
                content = hln[1:]
                # Find insertion point - this is simplified
                result.append(content)
            elif hln.startswith("-"):
                # Remove line
                content = hln[1:]
                # Skip this line
                orig_idx += 1
            elif hln.startswith(" "):
                # Context line - just advance
                orig_idx += 1
            else:
                # Might be a line number or other directive
                pass

        return None

    def _apply_simple_hunk(self, content: str, patch: str) -> Optional[str]:
        """Apply a simple patch in hunk format."""
        content_lines = content.splitlines(keepends=True)
        patch_lines = patch.splitlines()

        # Find the hunk start (@@ marker)
        start_idx = None
        for i, pl in enumerate(patch_lines):
            if pl.startswith("@@"):
                start_idx = i
                break

        if start_idx is None:
            # No unified diff format, try as direct replacement
            return self._apply_direct_replacement(content, patch)

        # Parse hunk: @@ -old_start,old_count +new_start,new_count @@
        hunk_header = patch_lines[start_idx]
        match = re.match(r"@@ -(\d+)(?:,(\d*))? \+(\d+)(?:,(\d*))? @@", hunk_header)
        if not match:
            return None

        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) else 1

        old_end = old_start + old_count - 1
        new_end = new_start + new_count - 1

        # Collect changes from the hunk
        added: List[str] = []
        removed: List[int] = []  # line numbers (1-indexed) to remove
        context: List[str] = []

        for i in range(start_idx + 1, len(patch_lines)):
            pl = patch_lines[i]
            if pl.startswith("@@"):
                # New hunk starting, stop here
                break
            elif pl.startswith("-"):
                removed.append(old_start + len(removed) + 1)  # 1-indexed
            elif pl.startswith("+"):
                added.append(pl[1:])
            elif pl.startswith(" "):
                context.append(pl[1:])
            else:
                # Unknown line, ignore
                pass

        # Apply changes
        new_content_lines: List[str] = []
        orig_idx = 0  # 0-indexed position in original

        for i, line in enumerate(content_lines):
            orig_idx = i + 1  # 1-indexed

            # Check if this line should be removed
            if orig_idx in removed:
                continue

            # Add context lines
            if orig_idx <= old_start or (orig_idx > old_end and not added):
                new_content_lines.append(line)

            # Add new lines from the patch
            for added_line in added:
                new_content_lines.append(added_line + "\n")

            # Add the original line if we're within the patched region
            if old_start <= orig_idx <= old_end:
                # Replace with corresponding new line
                new_line_idx = (orig_idx - old_start) + new_start - 1
                if new_line_idx < len(patch_lines) and patch_lines[start_idx + 1 + (orig_idx - old_start)].startswith("+"):
                    # Use the added line
                    pass
                new_content_lines.append(line)

        # This is getting complex - let's use a simpler approach
        # Just replace the old section with new content
        if new_count == 1 and len(added) == 1:
            # Simple single-line replacement
            new_line = added[0] if added else ""
            # Find the old line and replace
            if 1 <= old_start <= len(content_lines):
                new_lines = list(content_lines)
                new_lines[old_start - 1] = new_line + "\n" if not new_line.endswith("\n") else new_line
                return "".join(new_lines_lines)

        return None

    def _apply_direct_replacement(self, content: str, patch: str) -> Optional[str]:
        """Apply direct text replacement patch."""
        lines = content.splitlines(keepends=True)
        # If patch looks like it contains the full new content
        if patch.strip():
            return patch
        return None