"""Tests for internal tool representation and provider converters.

Internal rep (ToolRegistry.list_tools):
    {"name", "description", "parameter_schema"}

- OpenAI/OpenAI-compatible: {"type": "function",
    "function": {"name", "description", "parameters"}}
- Anthropic: {"name", "description", "input_schema"}

Also covers one tool-call round trip per provider into the internal
ToolCall representation consumed by agent.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

from minicoder.llm.tool_schemas import to_openai_tools, to_anthropic_tools
from minicoder.llm.openai import OpenAIClient
from minicoder.llm.openai_compatible import OpenAICompatibleClient
from minicoder.llm.anthropic import AnthropicClient
from minicoder.messages import Message, ToolCall
from minicoder.tools.finish import FinishTool
from minicoder.tools.registry import ToolRegistry


def _internal_tool() -> Dict[str, Any]:
    return {
        "name": "read_file",
        "description": "Read a file",
        "parameter_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }


class TestInternalRepresentation:
    def test_registry_returns_canonical_keys(self) -> None:
        registry = ToolRegistry()
        registry.register(FinishTool())
        tools = registry.list_tools()
        assert len(tools) == 1
        assert set(tools[0].keys()) == {"name", "description", "parameter_schema"}
        assert tools[0]["name"] == "finish"
        assert isinstance(tools[0]["parameter_schema"], dict)


class TestOpenAIConversion:
    def test_exact_openai_shape(self) -> None:
        converted = to_openai_tools([_internal_tool()])
        assert len(converted) == 1
        tool = converted[0]
        # Only documented OpenAI fields.
        assert set(tool.keys()) == {"type", "function"}
        assert tool["type"] == "function"
        assert set(tool["function"].keys()) == {"name", "description", "parameters"}
        assert tool["function"]["name"] == "read_file"
        assert tool["function"]["parameters"] == _internal_tool()["parameter_schema"]
        # Internal key must not leak.
        assert "parameter_schema" not in tool
        assert "input_schema" not in tool

    def test_empty_and_none(self) -> None:
        assert to_openai_tools([]) == []
        assert to_openai_tools(None) == []

    def test_message_args_serialized_as_json(self) -> None:
        client = OpenAIClient(model="m", api_key="k")
        tc = ToolCall(id="1", name="read_file", arguments={"path": ".", "recursive": True})
        built = client._build_openai_messages(
            [Message(role="assistant", content="", tool_calls=[tc])]
        )
        args_str = built[0]["tool_calls"][0]["function"]["arguments"]
        # Must be strict JSON (json.loads round-trips; str() repr would fail).
        assert json.loads(args_str) == {"path": ".", "recursive": True}


class TestAnthropicConversion:
    def test_exact_anthropic_shape(self) -> None:
        converted = to_anthropic_tools([_internal_tool()])
        assert len(converted) == 1
        tool = converted[0]
        # Only documented Anthropic fields.
        assert set(tool.keys()) == {"name", "description", "input_schema"}
        assert tool["name"] == "read_file"
        assert tool["input_schema"] == _internal_tool()["parameter_schema"]
        assert "parameter_schema" not in tool
        assert "parameters" not in tool
        assert "function" not in tool
        assert "type" not in tool

    def test_empty_and_none(self) -> None:
        assert to_anthropic_tools([]) == []
        assert to_anthropic_tools(None) == []

    def test_system_split_and_no_invented_fields(self) -> None:
        client = AnthropicClient.__new__(AnthropicClient)
        system, messages = client._build_anthropic_messages(
            [
                Message(role="system", content="sys-prompt"),
                Message(role="user", content="do it"),
                Message(role="tool", content="obs"),
            ]
        )
        assert system == "sys-prompt"
        assert all(m["role"] in ("user", "assistant") for m in messages)
        # Consecutive same-role messages merge so roles strictly alternate.
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "do it" in messages[0]["content"]
        assert "obs" in messages[0]["content"]
        # No invented provider fields in message payload.
        assert "tool_calls" not in messages[0]
        assert "system" not in [m.get("role") for m in messages]


class TestRoundTrip:
    def test_openai_round_trip(self) -> None:
        """internal -> OpenAI schema -> provider response -> ToolCall."""
        registry = ToolRegistry()
        registry.register(FinishTool())
        internal = registry.list_tools()
        wire = to_openai_tools(internal)
        assert wire[0]["function"]["name"] == "finish"

        args = {"summary": "done", "tests_run": 3, "remaining_issues": 0}
        client = OpenAIClient(model="m", api_key="k")
        fake_response = [
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="finish", arguments=json.dumps(args)),
            )
        ]
        parsed = client._parse_openai_tool_calls(fake_response)
        assert len(parsed) == 1
        assert parsed[0].name == "finish"
        assert parsed[0].arguments == args

        # Normalized call executes against the registry (agent.py path).
        tool = registry.lookup(parsed[0].name)
        assert tool is not None
        result = tool.execute(**parsed[0].arguments)
        assert result["summary"] == "done"

    def test_openai_compatible_round_trip(self) -> None:
        registry = ToolRegistry()
        registry.register(FinishTool())
        wire = to_openai_tools(registry.list_tools())
        assert wire[0]["type"] == "function"

        args = {"summary": "ok"}
        client = OpenAICompatibleClient(model="m", api_key="k")
        fake_response = [
            SimpleNamespace(
                id="call_9",
                function=SimpleNamespace(name="finish", arguments=json.dumps(args)),
            )
        ]
        parsed = client._parse_tool_calls(fake_response)
        assert parsed[0].arguments == args

    def test_anthropic_round_trip(self) -> None:
        """internal -> Anthropic schema -> tool_use blocks -> ToolCall."""
        registry = ToolRegistry()
        registry.register(FinishTool())
        internal = registry.list_tools()
        wire = to_anthropic_tools(internal)
        assert wire[0]["name"] == "finish"
        assert "input_schema" in wire[0]

        args = {"summary": "done"}
        client = AnthropicClient.__new__(AnthropicClient)
        obj_blocks = [SimpleNamespace(id="tu_1", name="finish", input=dict(args))]
        dict_blocks = [{"id": "tu_2", "type": "tool_use", "name": "finish", "input": dict(args)}]
        for blocks in (obj_blocks, dict_blocks):
            parsed = client._parse_anthropic_tool_calls(blocks)
            assert len(parsed) == 1
            assert parsed[0].name == "finish"
            assert parsed[0].arguments == args
            assert isinstance(parsed[0].arguments, dict)

        # Normalized call executes (agent.py path).
        tool = registry.lookup(parsed[0].name)
        assert tool is not None
        assert tool.execute(**parsed[0].arguments)["summary"] == "done"
