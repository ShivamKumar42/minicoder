"""Canonical internal tool representation and provider converters.

Internal representation (produced by ToolRegistry.list_tools(),
consumed by Agent and passed as `tools` to LLMClient.generate):

    {"name": str, "description": str, "parameter_schema": dict}
    where parameter_schema is a JSON Schema object, e.g.
    {"type": "object", "properties": {...}, "required": [...]}

Provider converters (deterministic, no invented fields):

- to_openai_tools: OpenAI Chat Completions format:
    {"type": "function",
     "function": {"name": ..., "description": ..., "parameters": {...}}}
- to_anthropic_tools: Anthropic Messages format:
    {"name": ..., "description": ..., "input_schema": {...}}
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


def to_openai_tools(
    internal_tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Convert internal tools to OpenAI Chat Completions `tools`."""
    converted: List[Dict[str, Any]] = []
    for tool in internal_tools or []:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameter_schema", {}),
                },
            }
        )
    return converted


def to_anthropic_tools(
    internal_tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Convert internal tools to Anthropic Messages `tools`."""
    converted: List[Dict[str, Any]] = []
    for tool in internal_tools or []:
        converted.append(
            {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameter_schema", {}),
            }
        )
    return converted
