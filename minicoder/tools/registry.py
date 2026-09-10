from __future__ import annotations
from typing import Any, Dict, Optional, List
from .base import Tool


class ToolRegistry:
    """Stores tools in the canonical internal representation.

    Each entry: {"name": str, "description": str,
                 "parameter_schema": JSON Schema dict}.
    Use llm.tool_schemas.to_openai_tools / to_anthropic_tools to
    convert to provider API formats before sending.
    """
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def lookup(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for name, tool in self._tools.items():
            result.append({
                "name": name,
                "description": tool.description,
                "parameter_schema": tool.parameter_schema,
            })
        return result

    def get_tool_schemas(self) -> Optional[List[Dict[str, Any]]]:
        """Get tool schemas for LLM consumption."""
        return self.list_tools()

    def get_schema_text(self) -> str:
        schemas: List[Dict[str, Any]] = self.list_tools()
        lines: List[str] = []
        for s in schemas:
            name = s["name"]
            desc = s["description"]
            params = s["parameter_schema"]
            lines.append(f"- {name}: {desc}")
            lines.append(f"  Parameters: {params}")
        return "\n".join(lines)