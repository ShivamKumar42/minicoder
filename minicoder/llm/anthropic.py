from __future__ import annotations
from typing import Any, List, Optional, Dict
import os

from .base import LLMClient
from ..messages import Message, ToolCall, ToolResult


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"

    @classmethod
    def supports_tools(cls) -> bool:
        return True

    def _build_anthropic_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        # Normalize to Anthropic format: system + user/assistant blocks
        result: List[Dict[str, Any]] = []
        system_block: Optional[Dict[str, Any]] = None

        for msg in messages:
            if msg.role == "system":
                system_block = {"role": "system", "content": msg.content}
            elif msg.role == "user":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                block: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    block["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "tool",
                            "function": {"name": tc.name, "arguments": str(tc.arguments)},
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(block)

        if system_block:
            result.insert(0, system_block)

        return result

    def _parse_anthropic_tool_calls(self, tool_calls: Any) -> List[ToolCall]:
        parsed: List[ToolCall] = []
        if tool_calls is None:
            return parsed
        # Anthropic returns tool_calls in a specific format
        for tc in tool_calls:
            parsed.append(
                ToolCall(
                    id=getattr(tc, 'id', str(tc.get('id', ''))),
                    name=getattr(tc, 'name', str(tc.get('name', ''))),
                    arguments=getattr(tc, 'input', str(tc.get('input', {}))),
                )
            )
        return parsed

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        anthropic_messages = self._build_anthropic_messages(messages)

        response = client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=anthropic_messages,
            tools=tools or [],
            tool_choice="auto",
        )

        tool_calls: List[ToolCall] = []
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'tool_use':
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )
                elif hasattr(block, 'type') and block.type == 'text':
                    pass  # content is in block.text

        return {
            "role": "assistant",
            "content": response.content[0].text if response.content else "",
            "tool_calls": tool_calls,
        }

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        anthropic_messages = self._build_anthropic_messages(messages)

        stream = client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=anthropic_messages,
            tools=tools or [],
            tool_choice="auto",
            stream=True,
        )

        return stream