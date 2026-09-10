"""Tests for the MiniCoder agent implementation.

These tests use the FakeLLMClient to verify the agent loop works
correctly without requiring API keys.
"""

from __future__ import annotations

import pytest

from minicoder.messages import Message, ToolCall, ToolResult
from minicoder.state import AgentState
from minicoder.tools.base import Tool
from minicoder.tools.registry import ToolRegistry
from minicoder.tools.filesystem import ListFilesTool
from minicoder.tools.read_file import ReadFileTool
from minicoder.tools.write_file import WriteFileTool
from minicoder.tools.apply_patch import ApplyPatchTool
from minicoder.tools.shell import RunCommandTool
from minicoder.tools.search import SearchFilesTool
from minicoder.tools.git import GitDiffTool, GitStatusTool
from minicoder.tools.finish import FinishTool
from minicoder.safety import SafetyTool
from minicoder.llm.base import LLMClient
from minicoder.llm.openai import OpenAIClient
from minicoder.llm.anthropic import AnthropicClient
from minicoder.llm.openai_compatible import OpenAICompatibleClient
from .fake_llm import FakeLLMClient
from minicoder.config import get_settings


class TestMessages:
    """Test the message format classes."""

    def test_message_creation(self) -> None:
        """Test basic Message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None

    def test_message_with_tool_calls(self) -> None:
        """Test Message with tool calls."""
        tc = ToolCall(id="call_1", name="test_tool", arguments={})
        msg = Message(role="assistant", content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "test_tool"

    def test_tool_call_creation(self) -> None:
        """Test ToolCall creation."""
        tc = ToolCall(id="call_123", name="list_files", arguments={"path": "."})
        assert tc.id == "call_123"
        assert tc.name == "list_files"
        assert tc.arguments == {"path": "."}

    def test_tool_result_creation(self) -> None:
        """Test ToolResult creation."""
        tr = ToolResult(tool_call_id="call_1", output="success", success=True)
        assert tr.tool_call_id == "call_1"
        assert tr.output == "success"
        assert tr.success is True


class TestState:
    """Test the AgentState structure."""

    def test_state_creation(self) -> None:
        """Test basic AgentState creation."""
        state = AgentState(task="test task", workspace="/tmp")
        assert state.task == "test task"
        assert state.workspace == "/tmp"
        assert state.finished is False
        assert state.current_iteration == 0

    def test_state_iteration(self) -> None:
        """Test iteration tracking."""
        state = AgentState(task="test", workspace="/tmp")
        assert state.can_proceed() is True
        state.current_iteration = 100
        assert state.can_proceed() is False  # max_iterations reached

    def test_state_error_tracking(self) -> None:
        """Test error tracking in state."""
        state = AgentState(task="test", workspace="/tmp")
        state.add_error("test error")
        assert "test error" in state.errors


class TestToolRegistry:
    """Test the tool registry."""

    def test_registry_creation(self) -> None:
        """Test basic registry creation."""
        registry = ToolRegistry()
        assert len(registry.list_tools()) == 0

    def test_register_and_lookup(self) -> None:
        """Test registering and looking up tools."""
        registry = ToolRegistry()

        class TestTool(Tool):
            name = "test_tool"
            description = "A test tool"

            def _build_schema(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            def execute(self, **kwargs: Any) -> Any:
                return "result"

        registry.register(TestTool())
        tool = registry.lookup("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"

    def test_list_tools_schema(self) -> None:
        """Test that list_tools returns proper schemas."""
        registry = ToolRegistry()

        class TestTool(Tool):
            name = "another_tool"
            description = "Another test"

            def _build_schema(self) -> dict[str, Any]:
                return {"type": "object", "properties": {"x": {"type": "int"}}}

            def execute(self, **kwargs: Any) -> Any:
                return kwargs.get("x", 0) * 2

        registry.register(TestTool())
        schemas = registry.list_tools()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "another_tool"
        assert schemas[0]["parameter_schema"]["properties"]["x"]["type"] == "int"


class TestFakeLLM:
    """Test the FakeLLMClient trajectory."""

    def test_fake_llm_trajectory(self) -> None:
        """Test that the fake LLM simulates a genuine coding-agent trajectory."""
        from minicoder.agent import Agent

        fake = FakeLLMClient(task="Fix the bugs in the demo project")
        registry = ToolRegistry()

        # Register all needed tools
        registry.register(ListFilesTool(workspace="/tmp"))
        registry.register(ReadFileTool(workspace="/tmp"))
        registry.register(WriteFileTool(workspace="/tmp"))
        registry.register(ApplyPatchTool(workspace="/tmp"))
        registry.register(RunCommandTool(workspace="/tmp"))
        registry.register(SearchFilesTool(workspace="/tmp"))
        registry.register(GitDiffTool(workspace="/tmp"))
        registry.register(GitStatusTool(workspace="/tmp"))
        registry.register(FinishTool())

        # Create a minimal agent
        # We can't fully test without a real workspace, but we can test
        # that the fake LLM produces the right trajectory steps

        # Just verify the fake LLM can generate responses
        messages = [Message(role="user", content="Fix bugs")]
        response = fake.generate(messages, tools=registry.get_tool_schemas())

        # First step of the trajectory must list files.
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0].name == "list_files"
        assert response["content"] != ""
        assert len(fake.call_history) == 1

    def test_fake_llm_steps(self) -> None:
        """Test that the fake LLM follows the expected steps."""
        fake = FakeLLMClient(task="Fix bugs")

        # Start with list_files
        messages = [Message(role="user", content="Fix bugs")]
        response = fake.generate(messages, tools=[])

        # Step should be 0 initially, then progress
        assert fake._step == 1  # After first generate

        # Generate multiple times to see steps progress
        for _ in range(10):
            response = fake.generate(messages)
            # Each call advances the step
        assert fake._step == 11  # Exactly one step per generate call


class TestConfig:
    """Test the configuration system."""

    def test_settings_loading(self) -> None:
        """Test that settings load from environment."""
        settings = get_settings()
        assert settings.provider in ("openai", "anthropic", "openai-compatible")
        assert settings.model != ""

    def test_env_override(self, monkeypatch) -> None:
        """Test environment variable overrides."""
        monkeypatch.setenv("MINICODER_PROVIDER", "anthropic")
        monkeypatch.setenv("MINICODER_MODEL", "claude-3-haiku-20240307")

        settings = get_settings()

        assert settings.provider == "anthropic"
        assert settings.model == "claude-3-haiku-20240307"


class TestSafety:
    """Test safety features."""

    def test_path_traversal_protection(self) -> None:
        """Test that path traversal is blocked."""
        from minicoder.safety import SafetyTool

        safety = SafetyTool(workspace="/workspace")

        # Test path traversal
        result = safety.execute("../../../etc/passwd")
        assert "escapes" in result.lower() or "security" in result.lower()

    def test_destructive_command_blocking(self) -> None:
        """Test that destructive commands are blocked."""
        from minicoder.safety import SafetyTool

        safety = SafetyTool(workspace="/workspace")

        result = safety.execute("rm -rf /")
        assert "blocked" in result.lower() or "security" in result.lower()

    def test_workspace_boundary(self) -> None:
        """Test that paths stay within workspace."""
        from minicoder.safety import SafetyTool

        safety = SafetyTool(workspace="/workspace")
        result = safety.execute("/etc/passwd")
        assert "escapes" in result.lower() or "security" in result.lower()


class TestPrompts:
    """Test the system prompt."""

    def test_system_prompt_contains_instructions(self) -> None:
        """Test that the system prompt contains core instructions."""
        from minicoder.prompts import SYSTEM_PROMPT

        assert "INSPECT BEFORE MODIFYING" in SYSTEM_PROMPT
        assert "SEARCH BEFORE GUESSING" in SYSTEM_PROMPT
        assert "apply_patch" in SYSTEM_PROMPT.lower()
        assert "finish" in SYSTEM_PROMPT.lower()


# Integration test with fake LLM
def test_agent_loop_with_fake_llm(tmp_path) -> None:
    """Integration test: run the agent loop with FakeLLMClient.

    Hermetic: the workspace contains the exact files the fake
    trajectory targets, and shell execution is stubbed, so no real
    subprocesses run and no hardcoded paths are needed.
    """
    import pytest

    from minicoder.agent import Agent
    from minicoder.tools.registry import ToolRegistry
    from minicoder.tools.filesystem import ListFilesTool
    from minicoder.tools.read_file import ReadFileTool
    from minicoder.tools.apply_patch import ApplyPatchTool
    from minicoder.tools.search import SearchFilesTool
    from minicoder.tools.finish import FinishTool
    from minicoder.state import AgentState

    workspace = tmp_path / "demo"
    (workspace / "app").mkdir(parents=True)
    (workspace / "tests").mkdir(parents=True)
    # Files the fake trajectory reads (inspection steps).
    (workspace / "app" / "calculator.py").write_text(
        'def add(a: int, b: int) -> int:\n    return a - b\n',
        encoding="utf-8",
    )
    (workspace / "app" / "validators.py").write_text(
        'def validate_password(password: str) -> bool:\n    return len(password) >= 3\n',
        encoding="utf-8",
    )
    (workspace / "tests" / "test_calculator.py").write_text(
        "def test_add():\n    assert True\n", encoding="utf-8"
    )
    (workspace / "tests" / "test_validators.py").write_text(
        "def test_validate():\n    assert True\n", encoding="utf-8"
    )
    # Files the fake trajectory patches (exact anchors it sends).
    (workspace / "app" / "calc_test.py").write_text(
        'def add(a: int, b: int) -> int:\n'
        '    """Add two numbers. BUG: subtracts instead of adding."""\n'
        '    return a - b\n',
        encoding="utf-8",
    )
    (workspace / "app" / "val_test.py").write_text(
        'def validate_password(password: str) -> bool:\n'
        '    """Validate password. BUG: accepts short passwords."""\n'
        '    if len(password) < 3:  # BUG: should be < 8\n'
        '        return False\n',
        encoding="utf-8",
    )

    class StubRunCommand(Tool):
        name = "run_command"
        description = "Stubbed shell for tests"

        def _build_schema(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        def execute(self, **kwargs: Any) -> Any:
            return "3 passed"

    fake = FakeLLMClient(task="Fix bugs in calculator and validators")
    registry = ToolRegistry()

    # Register all tools
    registry.register(ListFilesTool(workspace=str(workspace)))
    registry.register(ReadFileTool(workspace=str(workspace)))
    registry.register(ApplyPatchTool(workspace=str(workspace)))
    registry.register(StubRunCommand())
    registry.register(SearchFilesTool(workspace=str(workspace)))
    registry.register(FinishTool())

    # Create agent
    agent = Agent(
        llm_client=fake,
        registry=registry,
        task="Fix bugs in calculator and validators",
        workspace=str(workspace),
        max_iterations=20,
        max_tool_calls=50,
        approval_mode="auto",
        dry_run=False,
        verbose=False,
    )

    # Run the agent
    result = agent.run()

    # Verify results: both patches applied, reported, and no errors.
    assert result["finished"] is True
    assert result["iterations"] > 0
    assert result["tool_calls"] > 0
    assert result["final_response"]
    assert result["errors"] == []
    assert sorted(result["modified_files"]) == ["app/calc_test.py", "app/val_test.py"]
    assert "return a + b" in (workspace / "app" / "calc_test.py").read_text(encoding="utf-8")
    assert "len(password) < 8" in (workspace / "app" / "val_test.py").read_text(encoding="utf-8")