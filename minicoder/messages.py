from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, List


@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    output: Optional[str] = None
    success: bool = True
    metadata: Optional[dict[str, Any]] = None