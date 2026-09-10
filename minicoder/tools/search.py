from __future__ import annotations
import re
import os
import fnmatch
from typing import Dict, Any, Optional, List
from .base import Tool
from .budget import (
    DEFAULT_MAX_MATCHES_PER_FILE,
    DEFAULT_MAX_MATCH_LINE_CHARS,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_MAX_SEARCH_FILES,
    DEFAULT_MAX_SEARCH_FILE_BYTES,
    DEFAULT_MAX_SEARCH_TOTAL_BYTES,
    truncate_text,
)


class SearchFilesTool(Tool):
    def __init__(
        self,
        workspace: str = ".",
        max_files: int = DEFAULT_MAX_SEARCH_FILES,
        max_matches_per_file: int = DEFAULT_MAX_MATCHES_PER_FILE,
        max_line_chars: int = DEFAULT_MAX_MATCH_LINE_CHARS,
        max_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        max_file_bytes: int = DEFAULT_MAX_SEARCH_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_SEARCH_TOTAL_BYTES,
    ):
        super().__init__()
        self.name = "search_files"
        self.description = "Search file contents for a text or regex pattern"
        self.workspace = os.path.abspath(workspace)
        self.max_files = max_files
        self.max_matches_per_file = max_matches_per_file
        self.max_line_chars = max_line_chars
        self.max_chars = max_chars
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text pattern or regex to search for"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Whether to ignore case (default: true)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file pattern to limit search (e.g., '*.py')"
                }
            },
            "required": ["pattern"]
        }

    def _search_in_file(self, content: str, pattern: str, case_insensitive: bool = True) -> List[Dict[str, Any]]:
        flags = re.IGNORECASE if case_insensitive else 0
        results: List[Dict[str, Any]] = []
        for line_num, line in enumerate(content.splitlines(), 1):
            if re.search(pattern, line, flags):
                match = re.search(pattern, line, flags)
                results.append({
                    "line": line_num,
                    "content": line,
                    "match": match.group() if match else ""
                })
        return results

    def execute(self, pattern: str, case_insensitive: bool = True,
                file_pattern: Optional[str] = None) -> str:
        workspace = self.workspace

        results: List[Dict[str, Any]] = []
        skipped_large = 0

        # Collect candidates first so scanning order is deterministic
        # (sorted); the byte budget below then stops at a stable point.
        candidates: List[str] = []
        for root, dirs, files in os.walk(workspace):
            ignored = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
            dirs[:] = [d for d in dirs if d not in ignored]

            for fname in files:
                if file_pattern and not fnmatch.fnmatch(fname, file_pattern):
                    continue
                candidates.append(os.path.join(root, fname))
        candidates.sort(key=lambda p: os.path.relpath(p, workspace))

        scanned = 0
        scanned_bytes = 0
        budget_exhausted = False
        for full_path in candidates:
            if scanned_bytes >= self.max_total_bytes:
                budget_exhausted = True
                break
            try:
                if os.path.getsize(full_path) > self.max_file_bytes:
                    skipped_large += 1
                    continue
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                scanned += 1
                scanned_bytes += len(content)
                file_results = self._search_in_file(content, pattern, case_insensitive)
                if file_results:
                    results.append({
                        "file": os.path.relpath(full_path, workspace),
                        "matches": file_results
                    })
            except Exception:
                continue

        if not results:
            if budget_exhausted:
                return (
                    f"No matches found for pattern: {pattern} "
                    f"(stopped after scanning {scanned} of "
                    f"{len(candidates)} files: size budget exhausted; "
                    f"narrow file_pattern)"
                )
            if skipped_large:
                return (
                    f"No matches found for pattern: {pattern} "
                    f"(skipped {skipped_large} files over size limit)"
                )
            return f"No matches found for pattern: {pattern}"

        results.sort(key=lambda r: r["file"])
        output_parts: List[str] = []
        shown_files = results[: self.max_files]
        for r in shown_files:
            output_parts.append(f"File: {r['file']}")
            shown_matches = r["matches"][: self.max_matches_per_file]
            for m in shown_matches:
                line = m["content"]
                if len(line) > self.max_line_chars:
                    line = line[: self.max_line_chars] + "…"
                output_parts.append(f"  Line {m['line']}: {line}")
            omitted_matches = len(r["matches"]) - len(shown_matches)
            if omitted_matches > 0:
                output_parts.append(
                    f"  ... [{omitted_matches} more matches in {r['file']} omitted]"
                )
        omitted_files = len(results) - len(shown_files)
        if omitted_files > 0:
            output_parts.append(
                f"... [{omitted_files} more files with matches omitted; "
                f"narrow file_pattern to see them]"
            )
        if skipped_large > 0:
            output_parts.append(
                f"... [skipped {skipped_large} files over size limit]"
            )
        if budget_exhausted:
            output_parts.append(
                f"... [stopped after scanning {scanned} of "
                f"{len(candidates)} files (size budget); "
                f"narrow file_pattern to see the rest]"
            )

        output = "\n".join(output_parts)
        output, _ = truncate_text(
            output,
            self.max_chars,
            hint="(narrow pattern/file_pattern to see the rest)",
        )
        return output