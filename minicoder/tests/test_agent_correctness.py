"""Focused regression tests for agent task delivery, response parsing,
and completion logic.

Covers ONLY:
1. Task/workspace is included in LLM messages sent to generate().
2. Both dict-style and object-style responses are parsed.
3. Plain text responses become the final answer (not discarded).
4. Tool-call responses continue the loop (do not finish prematurely).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from minicoder.agent import Agent
from minicoder.messages import Message, ToolCall
from minicoder.tools.base import Tool
from minicoder.tools.registry import ToolRegistry
from minicoder.tools.finish import FinishTool


class CapturingLLM:
    """Records messages passed to generate() and returns a fixed response."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.seen_messages: Optional[List[Message]] = None

    def generate(self, messages: List[Message], tools: Any = None) -> Any:
        self.seen_messages = list(messages)
        return self.response


class EchoTool(Tool):
    name = "echo_tool"
    description = "Echo tool for tests"

    def _build_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        return "echo-ok"


def _make_agent(llm_client: Any, task: str = "Fix the login bug", workspace: str = "/tmp") -> Agent:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(FinishTool())
    return Agent(
        llm_client=llm_client,
        registry=registry,
        task=task,
        workspace=workspace,
        max_iterations=5,
        max_tool_calls=10,
    )


class TestTaskDelivery:
    def test_initial_state_contains_task(self) -> None:
        agent = _make_agent(CapturingLLM({"content": "done", "tool_calls": []}))
        contents = [m.content for m in agent.state.messages if m.role == "user"]
        assert any("Fix the login bug" in c for c in contents)

    def test_generate_request_contains_task_and_workspace(self) -> None:
        llm = CapturingLLM({"content": "done", "tool_calls": []})
        agent = _make_agent(llm, task="Fix the login bug", workspace="/tmp")
        agent._generate_response()
        assert llm.seen_messages is not None
        joined = "\n".join(m.content for m in llm.seen_messages)
        assert "Fix the login bug" in joined
        assert "/tmp" in joined


class TestResponseParsing:
    def test_dict_plain_text_becomes_final(self) -> None:
        agent = _make_agent(CapturingLLM({}))
        agent._process_response({"content": "all done", "tool_calls": []})
        assert agent.state.finished is True
        assert agent.state.final_response == "all done"

    def test_object_plain_text_becomes_final(self) -> None:
        agent = _make_agent(CapturingLLM({}))
        agent._process_response(SimpleNamespace(content="object done", tool_calls=[]))
        assert agent.state.finished is True
        assert agent.state.final_response == "object done"

    def test_dict_text_key_fallback(self) -> None:
        agent = _make_agent(CapturingLLM({}))
        agent._process_response({"text": "fallback text"})
        assert agent.state.finished is True
        assert agent.state.final_response == "fallback text"

    def test_object_text_attr_fallback(self) -> None:
        agent = _make_agent(CapturingLLM({}))
        agent._process_response(SimpleNamespace(text="obj fallback", tool_calls=None))
        assert agent.state.finished is True
        assert agent.state.final_response == "obj fallback"

    def test_dict_tool_calls_continue_loop(self) -> None:
        agent = _make_agent(CapturingLLM({}))
        tc = {"id": "1", "name": "echo_tool", "arguments": {}}
        agent._process_response({"content": "working", "tool_calls": [tc]})
        # Tool-call response must NOT finish; it continues the loop.
        assert agent.state.finished is False
        assert agent.total_tool_calls == 1
        # Tool observation must be recorded, content must not be lost silently.
        assert any("echo-ok" in m.content for m in agent.state.messages)

    def test_object_tool_calls_continue_loop(self) -> None:
        agent = _make_agent(CapturingLLM({}))
        tc = ToolCall(id="2", name="echo_tool", arguments={})
        agent._process_response(SimpleNamespace(content="working", tool_calls=[tc]))
        assert agent.state.finished is False
        assert agent.total_tool_calls == 1

    def test_empty_response_not_silently_discarded(self) -> None:
        agent = _make_agent(CapturingLLM({}))
        agent._process_response({"content": "", "tool_calls": []})
        assert agent.state.finished is False
        assert len(agent.state.errors) == 1


class TestCompletionEndToEnd:
    def test_run_with_plain_text_llm_finishes_with_answer(self) -> None:
        llm = CapturingLLM({"content": "task complete", "tool_calls": []})
        agent = _make_agent(llm)
        result = agent.run()
        assert result["finished"] is True
        assert result["final_response"] == "task complete"
        assert result["iterations"] == 1
