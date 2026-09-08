from __future__ import annotations
import re
import os
import fnmatch
from typing import Dict, Any, Optional, List, Tuple
from .base import Tool


class SearchFilesTool(Tool):
    def __init__(self, workspace: str = "."):
        super().__init__()
        self.name = "search_files"
        self.workspace = os.path.abspath(workspace)

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
        compiled_pattern = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)

        for root, dirs, files in os.walk(workspace):
            ignored = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
            dirs[:] = [d for d in dirs if d not in ignored]

            for fname in files:
                if file_pattern and not fnmatch.fnmatch(fname, file_pattern):
                    continue

                full_path = os.path.join(root, fname)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    file_results = self._search_in_file(content, pattern, case_insensitive)
                    if file_results:
                        results.append({
                            "file": os.path.relpath(full_path, workspace),
                            "matches": file_results
                        })
                except Exception:
                    continue

        if not results:
            return f"No matches found for pattern: {pattern}"

        output_parts: List[str] = []
        for r in results[:50]:
            output_parts.append(f"File: {r['file']}")
            for m in r["matches"][:5]:
                output_parts.append(f"  Line {m['line']}: {m['content'][:100]}")

        return "\n".join(output_parts)