from __future__ import annotations
from typing import Any, List, Optional, Dict
import os

from .base import LLMClient
from .tool_schemas import to_anthropic_tools
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

    def _build_anthropic_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Split internal messages into Anthropic (system, messages).

        Anthropic takes `system` as a separate top-level parameter and
        `messages` with only user/assistant roles in strict alternation.
        Internal `tool` messages are forwarded as user context, and
        consecutive same-role messages are merged so the sequence stays
        valid (no invented fields).
        """
        system_parts: List[str] = []
        merged: List[Dict[str, Any]] = []

        def _push(role: str, content: str) -> None:
            if merged and merged[-1]["role"] == role:
                merged[-1]["content"] += "\n\n" + content
            else:
                merged.append({"role": role, "content": content})

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "assistant":
                _push("assistant", msg.content or "")
            elif msg.role in ("user", "tool"):
                _push("user", msg.content)
            else:
                _push("user", msg.content)

        system = "\n".join(system_parts) if system_parts else None
        return system, merged

    def _parse_anthropic_tool_calls(self, tool_calls: Any) -> List[ToolCall]:
        """Normalize Anthropic tool_use blocks to internal ToolCall."""
        parsed: List[ToolCall] = []
        if tool_calls is None:
            return parsed
        for tc in tool_calls:
            if isinstance(tc, dict):
                tc_id = str(tc.get("id", ""))
                tc_name = str(tc.get("name", ""))
                tc_input = tc.get("input", {})
            else:
                tc_id = str(getattr(tc, "id", ""))
                tc_name = str(getattr(tc, "name", ""))
                tc_input = getattr(tc, "input", {})
            if not isinstance(tc_input, dict):
                tc_input = {}
            parsed.append(ToolCall(id=tc_id, name=tc_name, arguments=tc_input))
        return parsed

    @staticmethod
    def _extract_anthropic_text(content: Any) -> str:
        """Join all text blocks; ignore tool_use blocks."""
        texts: List[str] = []
        for block in content or []:
            block_type = (
                block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            )
            if block_type != "text":
                continue
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
            if isinstance(text, str) and text:
                texts.append(text)
        return "".join(texts)

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        system, anthropic_messages = self._build_anthropic_messages(messages)
        if not anthropic_messages:
            raise ValueError("No messages to send to the model")

        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        converted_tools = to_anthropic_tools(tools)
        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
            "tools": converted_tools,
        }
        if converted_tools:
            create_kwargs["tool_choice"] = "auto"
        if system:
            create_kwargs["system"] = system

        response = client.messages.create(**create_kwargs)

        tool_use_blocks: List[Any] = []
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                block_type = getattr(block, 'type', None)
                if isinstance(block, dict):
                    block_type = block.get('type')
                if block_type == 'tool_use':
                    tool_use_blocks.append(block)

        tool_calls = self._parse_anthropic_tool_calls(tool_use_blocks)
        content = self._extract_anthropic_text(
            response.content if hasattr(response, 'content') else []
        )

        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        system, anthropic_messages = self._build_anthropic_messages(messages)

        converted_tools = to_anthropic_tools(tools)
        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
            "tools": converted_tools,
            "stream": True,
        }
        if converted_tools:
            create_kwargs["tool_choice"] = "auto"
        if system:
            create_kwargs["system"] = system

        stream = client.messages.create(**create_kwargs)

        return stream