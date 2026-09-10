from __future__ import annotations
from typing import Any, ClassVar, List, Optional, Dict

from .openai_base import OpenAIChatBase
from ..messages import Message, ToolCall


class OpenAICompatibleClient(OpenAIChatBase):
    _PROVIDER_NAME: ClassVar[str] = "openai-compatible"
    _ENV_API_KEY: ClassVar[str] = "OPENAI_COMPATIBLE_API_KEY"
    _ENV_BASE_URL: ClassVar[Optional[str]] = "OPENAI_COMPATIBLE_BASE_URL"
    _DEFAULT_BASE_URL: ClassVar[Optional[str]] = "http://localhost:11434/v1"

    # Backwards-compatible aliases for the historical method names.
    def _build_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        return self._build_openai_messages(messages)

    def _parse_tool_calls(self, tool_calls: Any) -> List[ToolCall]:
        return self._parse_openai_tool_calls(tool_calls)
