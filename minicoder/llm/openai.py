from __future__ import annotations
from typing import Any, List, Optional, Dict
import os

from .base import LLMClient
from ..messages import Message, ToolCall, ToolResult


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    @classmethod
    def supports_tools(cls) -> bool:
        return True

    def _build_openai_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for msg in messages:
            openai_msg: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in msg.tool_calls
                ]
            result.append(openai_msg)
        return result

    def _parse_openai_tool_calls(
        self, tool_calls: Any
    ) -> List[ToolCall]:
        parsed: List[ToolCall] = []
        if tool_calls is None:
            return parsed
        for tc in tool_calls:
            parsed.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=eval(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                )
            )
        return parsed

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        import openai

        client = openai.OpenAI(
            api_key=self._api_key, base_url=self._base_url
        )

        openai_messages = self._build_openai_messages(messages)

        response = client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            tools=tools,
        )

        if response.choices[0].message.tool_calls:
            tool_calls = self._parse_openai_tool_calls(
                response.choices[0].message.tool_calls
            )
        else:
            tool_calls = []

        return {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content or "",
            "tool_calls": tool_calls,
        }

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        import openai

        client = openai.OpenAI(
            api_key=self._api_key, base_url=self._base_url
        )

        openai_messages = self._build_openai_messages(messages)

        stream = client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            tools=tools,
            stream=True,
        )

        return stream