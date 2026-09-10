"""Tests for AgentState session persistence (save -> load -> continue)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from minicoder.agent import Agent
from minicoder.messages import Message, ToolCall
from minicoder.state import AgentState
from minicoder.tools.base import Tool
from minicoder.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo_tool"
    description = "Echo tool for tests"

    def _build_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        return "echo-ok"


class ScriptedLLM:
    """Returns queued responses in order (repeats the last one)."""

    def __init__(self, responses: List[Any]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, messages: Any, tools: Any = None) -> Any:
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


def _full_state() -> AgentState:
    state = AgentState(task="Fix it", workspace="/tmp")
    state.add_message(Message(role="user", content="Task: Fix it"))
    state.add_message(
        Message(
            role="assistant",
            content="working",
            tool_calls=[ToolCall(id="1", name="echo_tool", arguments={})],
        )
    )
    state.add_message(Message(role="tool", content="echo-ok"))
    state.current_iteration = 3
    state.tool_calls_executed.append(
        ToolCall(id="1", name="echo_tool", arguments={})
    )
    state.modified_files.append("a.py")
    state.test_results = {"test_a": True}
    state.errors.append("some error")
    state.final_response = None
    return state


class TestSaveLoad:
    def test_round_trip_preserves_everything(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = str(tmp_path / "session.json")
        state = _full_state()
        state.save(path, total_tool_calls=7)

        loaded, total = AgentState.load(path)
        assert total == 7
        assert loaded.task == "Fix it"
        assert loaded.workspace == "/tmp"
        assert loaded.current_iteration == 3
        assert loaded.max_iterations == state.max_iterations
        assert len(loaded.messages) == 3
        assert loaded.messages[1].tool_calls is not None
        assert loaded.messages[1].tool_calls[0].name == "echo_tool"
        assert loaded.messages[2].content == "echo-ok"
        assert len(loaded.tool_calls_executed) == 1
        assert loaded.modified_files == ["a.py"]
        assert loaded.test_results == {"test_a": True}
        assert loaded.errors == ["some error"]
        assert loaded.finished is False
        assert loaded.final_response is None

    def test_no_secrets_persisted(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = str(tmp_path / "session.json")
        _full_state().save(path)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        assert "api_key" not in raw.lower()
        assert "OPENAI_API_KEY" not in raw
        # to_dict output itself must be JSON-serializable without secrets.
        assert json.loads(raw)["state"]["task"] == "Fix it"

    def test_missing_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(FileNotFoundError):
            AgentState.load(str(tmp_path / "nope.json"))

    def test_corrupted_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="[Cc]orrupt"):
            AgentState.load(str(path))

    def test_invalid_shape(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "bad.json"
        path.write_text('{"version": 1}', encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'state'"):
            AgentState.load(str(path))


class TestSaveLoadContinue:
    def test_save_load_continue(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        session = str(tmp_path / "session.json")
        tool_response = {
            "content": "step one",
            "tool_calls": [ToolCall(id="1", name="echo_tool", arguments={})],
        }
        agent = Agent(
            llm_client=ScriptedLLM([tool_response]),
            registry=_registry(),
            task="Do things",
            workspace=str(tmp_path),
        )
        # Simulate one interrupted iteration (tool executed, no finish).
        agent._process_response(agent.llm_client.generate([]))
        agent.state.current_iteration = 1
        assert agent.total_tool_calls == 1
        agent.save_session(session)

        # Resume with a fresh agent: plain-text answer finishes the run.
        resumed = Agent.from_session(
            ScriptedLLM([{"content": "all done", "tool_calls": []}]),
            _registry(),
            session,
        )
        assert resumed.total_tool_calls == 1
        assert any("echo-ok" in m.content for m in resumed.state.messages)
        result = resumed.run()
        assert result["finished"] is True
        assert result["final_response"] == "all done"
        # Iteration count continued instead of restarting at 1.
        assert result["iterations"] == 2
        assert result["tool_calls"] == 1
        # History from before the interruption survived.
        assert any("echo-ok" in m.content for m in resumed.state.messages)


class TestSessionCLI:
    def test_flags_parsed(self) -> None:
        from minicoder.cli import CLI

        parsed = CLI().parse_args(
            ["my task", "--session", "s.json", "--resume"]
        )
        assert parsed.session == "s.json"
        assert parsed.resume is True
        assert parsed.task == "my task"

    def test_resume_requires_session(self, capsys) -> None:  # type: ignore[no-untyped-def]
        from minicoder.cli import CLI

        assert CLI().run(["--resume"]) == 1
        assert "--session" in capsys.readouterr().out

    def test_resume_missing_file_is_graceful(self) -> None:
        from minicoder.cli import CLI

        cli = CLI()
        result = cli._run_agent(
            "", session="/nonexistent-minicoder-session.json", resume=True
        )
        assert result["finished"] is False
        assert any("Cannot resume" in e for e in result["errors"])
