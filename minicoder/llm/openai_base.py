"""Shared Chat-Completions implementation for OpenAI-API-compatible clients.

OpenAIClient and OpenAICompatibleClient differ only in provider name,
default API key env var, and default base URL. Everything else
(message building, tool-call parsing, generate/stream) is identical
and lives here so the two subclasses cannot drift apart.
"""

from __future__ import annotations
from typing import Any, ClassVar, List, Optional, Dict
import json
import os

from .base import LLMClient
from .tool_schemas import to_openai_tools
from ..messages import Message, ToolCall, ToolResult


class OpenAIChatBase(LLMClient):
    _PROVIDER_NAME: ClassVar[str] = "openai"
    _ENV_API_KEY: ClassVar[str] = "OPENAI_API_KEY"
    _ENV_BASE_URL: ClassVar[Optional[str]] = None
    _DEFAULT_BASE_URL: ClassVar[Optional[str]] = None

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get(self._ENV_API_KEY, "")
        if self._ENV_BASE_URL is None:
            self._base_url = base_url
        else:
            self._base_url = base_url or os.environ.get(
                self._ENV_BASE_URL, self._DEFAULT_BASE_URL or ""
            )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._PROVIDER_NAME

    @classmethod
    def supports_tools(cls) -> bool:
        return True

    def _build_openai_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for index, msg in enumerate(messages):
            if msg.role == "tool":
                # OpenAI requires tool_call_id on tool messages; fall back
                # to the recorded call id, then to a deterministic placeholder.
                call_id = msg.tool_call_id
                if not call_id and msg.tool_calls:
                    call_id = msg.tool_calls[0].id
                openai_msg = {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": call_id or f"legacy_{index}",
                }
                result.append(openai_msg)
                continue
            openai_msg: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
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
            raw_args = tc.function.arguments
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid tool arguments JSON for tool "
                        f"'{tc.function.name}': {e}"
                    ) from e
                if not isinstance(arguments, dict):
                    raise ValueError(
                        f"Tool arguments JSON must decode to an object for tool "
                        f"'{tc.function.name}', got: {type(arguments).__name__}"
                    )
            else:
                arguments = raw_args
            parsed.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )
        return parsed

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        openai_messages = self._build_openai_messages(messages)
        if not openai_messages:
            raise ValueError("No messages to send to the model")

        import openai

        client = openai.OpenAI(
            api_key=self._api_key, base_url=self._base_url
        )

        response = client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            tools=to_openai_tools(tools),
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
            tools=to_openai_tools(tools),
            stream=True,
        )

        return stream
