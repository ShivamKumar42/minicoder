from __future__ import annotations
from typing import Dict, Any, Optional
from .base import Tool


class FinishTool(Tool):
    def __init__(self):
        super().__init__()
        self.name = "finish"
        self.description = "Explicitly tell the agent that the task is complete. Provide a summary of what was accomplished, tests run, and any remaining issues."

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary of what was accomplished",
                },
                "tests_run": {
                    "type": "integer",
                    "description": "Number of tests run",
                },
                "remaining_issues": {
                    "type": "integer",
                    "description": "Number of remaining issues, if any"
                }
            },
            "required": ["summary"],
        }

    def execute(
        self,
        summary: str,
        tests_run: int = 0,
        remaining_issues: int = 0,
    ) -> dict[str, Any]:
        return {
            "summary": summary,
            "tests_run": tests_run,
            "remaining_issues": remaining_issues,
        }