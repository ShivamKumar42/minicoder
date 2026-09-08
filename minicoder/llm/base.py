from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
from ..messages import Message, ToolCall, ToolResult


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        pass

    @classmethod
    @abstractmethod
    def supports_tools(cls) -> bool:
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        pass