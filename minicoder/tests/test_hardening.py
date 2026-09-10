"""Hardening tests for adversarial-review fixes.

Covers: bounded LLM-error retries, max_iterations propagation, honest
`finished`, malformed tool_calls, assistant/tool_call_id history,
provider payload validity, dry-run, approval recording + word
boundaries, list_files confinement, modified_files/errors visibility,
session validation, history cap, SafetyTool registration, git paths,
search size-skip, read paging, CLI flags.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from minicoder.agent import Agent
from minicoder.messages import Message, ToolCall
from minicoder.state import AgentState
from minicoder.tools.base import Tool
from minicoder.tools.registry import ToolRegistry
from minicoder.tools.finish import FinishTool


class EchoTool(Tool):
    name = "echo_tool"
    description = "Echo"

    def _build_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        return "echo-ok"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(FinishTool())
    return reg


class TestLoopBounds:
    def test_always_failing_llm_terminates(self) -> None:
        class AlwaysFails:
            def generate(self, messages: Any, tools: Any = None) -> Any:
                raise RuntimeError("boom")

        agent = Agent(
            llm_client=AlwaysFails(),
            registry=_registry(),
            task="t",
            workspace="/tmp",
            max_iterations=100,
        )
        result = agent.run()
        assert result["finished"] is False
        assert result["iterations"] <= 100
        assert any("Aborting after 3 consecutive" in e for e in result["errors"])

    def test_max_iterations_propagated(self) -> None:
        class Quiet:
            def generate(self, messages: Any, tools: Any = None) -> Any:
                return {"content": "x", "tool_calls": [{"id": "1", "name": "echo_tool", "arguments": {}}]}

        agent = Agent(
            llm_client=Quiet(),
            registry=_registry(),
            task="t",
            workspace="/tmp",
            max_iterations=5,
            max_tool_calls=1000,
        )
        assert agent.state.max_iterations == 5
        result = agent.run()
        assert result["iterations"] <= 5
        assert result["finished"] is False  # limit hit, not completion

    def test_malformed_tool_calls_do_not_crash(self) -> None:
        agent = Agent(
            llm_client=None, registry=_registry(), task="t", workspace="/tmp"
        )
        agent._process_response({"content": "", "tool_calls": {"a": 1}})
        assert agent.state.finished is False
        assert any("Malformed" in e for e in agent.state.errors)
        agent._process_response({"content": "", "tool_calls": [None]})
        assert agent.state.finished is False


class TestHistoryValidity:
    def _run_one_tool(self) -> Agent:
        class OneTool:
            def generate(self, messages: Any, tools: Any = None) -> Any:
                return {
                    "content": "go",
                    "tool_calls": [ToolCall(id="abc", name="echo_tool", arguments={})],
                }

        agent = Agent(llm_client=OneTool(), registry=_registry(), task="t", workspace="/tmp")
        agent._process_response(agent.llm_client.generate([]))
        return agent

    def test_assistant_request_and_ids_stored(self) -> None:
        agent = self._run_one_tool()
        roles = [m.role for m in agent.state.messages]
        assert roles == ["user", "assistant", "tool"]
        assert agent.state.messages[1].tool_calls is not None
        assert agent.state.messages[1].tool_calls[0].id == "abc"
        assert agent.state.messages[2].tool_call_id == "abc"

    def test_openai_payload_has_ids(self) -> None:
        from minicoder.llm.openai import OpenAIClient

        agent = self._run_one_tool()
        payload = OpenAIClient(model="m", api_key="k")._build_openai_messages(
            agent.state.messages
        )
        by_role = {m["role"]: m for m in payload}
        assert by_role["tool"]["tool_call_id"] == "abc"
        assistant = [m for m in payload if m["role"] == "assistant"][0]
        assert assistant["tool_calls"][0]["id"] == "abc"

    def test_anthropic_roles_alternate(self) -> None:
        from minicoder.llm.anthropic import AnthropicClient

        agent = self._run_one_tool()
        client = AnthropicClient.__new__(AnthropicClient)
        _, msgs = client._build_anthropic_messages(
            [Message(role="system", content="S")] + agent.state.messages
        )
        assert msgs[0]["role"] == "user"
        for prev, cur in zip(msgs, msgs[1:]):
            assert prev["role"] != cur["role"]


class TestDryRunApproval:
    def test_dry_run_skips_mutations(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from minicoder.tools.write_file import WriteFileTool

        reg = ToolRegistry()
        reg.register(WriteFileTool(workspace=str(tmp_path)))
        reg.register(EchoTool())

        class Writer:
            def generate(self, messages: Any, tools: Any = None) -> Any:
                return {
                    "content": "w",
                    "tool_calls": [
                        ToolCall(id="1", name="write_file", arguments={"path": "a.txt", "content": "hi"})
                    ],
                }

        agent = Agent(
            llm_client=Writer(), registry=reg, task="t",
            workspace=str(tmp_path), dry_run=True,
        )
        agent._process_response(agent.llm_client.generate([]))
        assert not (tmp_path / "a.txt").exists()
        # Skipped calls still count as attempts (bounds the loop).
        assert agent.total_tool_calls == 1
        assert any("Dry run" in m.content for m in agent.state.messages)

    def test_approval_denial_recorded(self) -> None:
        from minicoder.tools.shell import RunCommandTool

        reg = ToolRegistry()
        reg.register(RunCommandTool(workspace="/tmp"))
        agent = Agent(
            llm_client=None, registry=reg, task="t",
            workspace="/tmp", approval_mode="deny-dangerous",
        )
        agent._execute_tool_call(
            ToolCall(id="1", name="run_command", arguments={"command": "rm -rf /"})
        )
        assert any("denied" in e.lower() for e in agent.state.errors)
        assert any("denied" in m.content.lower() for m in agent.state.messages)

    def test_word_boundaries(self) -> None:
        from minicoder.tools.shell import RunCommandTool
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tool = RunCommandTool(workspace=d)
            assert tool._is_blocked("sudo ls") is True
            assert tool._is_blocked("echo address") is False  # "dd" substring
            assert tool._is_blocked("echo hello") is False
        agent = Agent(
            llm_client=None, registry=_registry(), task="t",
            workspace="/tmp", approval_mode="deny-dangerous",
        )
        assert agent._check_approval("run_command", {"command": "echo formatting"}) is True
        assert agent._check_approval("run_command", {"command": "rm -rf /"}) is False


class TestVisibility:
    def test_tool_error_recorded_and_modified_tracked(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from minicoder.tools.write_file import WriteFileTool

        reg = ToolRegistry()
        reg.register(WriteFileTool(workspace=str(tmp_path)))

        agent = Agent(llm_client=None, registry=reg, task="t", workspace=str(tmp_path))
        agent._execute_tool_call(
            ToolCall(id="1", name="write_file", arguments={"path": "a.txt", "content": "hi"})
        )
        assert "a.txt" in agent.state.modified_files
        agent._execute_tool_call(
            ToolCall(id="2", name="write_file", arguments={"path": "../evil.txt", "content": "x"})
        )
        assert any("failed" in e for e in agent.state.errors)

    def test_list_files_escape_blocked(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from minicoder.tools.filesystem import ListFilesTool

        out = ListFilesTool(str(tmp_path)).execute(path="../..")
        assert "escapes workspace" in out

    def test_history_bounded(self) -> None:
        agent = Agent(
            llm_client=None, registry=_registry(), task="t",
            workspace="/tmp", max_history=10,
        )
        for i in range(30):
            agent._append_message(Message(role="tool", content=f"m{i}"))
        assert len(agent.state.messages) <= 10
        assert agent.state.messages[0].role == "user"  # seed preserved


class TestSessionValidation:
    def test_bad_ints_rejected_as_value_error(self) -> None:
        with pytest.raises(ValueError):
            AgentState.from_dict({"task": "t", "workspace": "w", "current_iteration": None})
        with pytest.raises(ValueError):
            AgentState.from_dict({"task": "t", "workspace": "w", "modified_files": None})
        with pytest.raises(ValueError):
            AgentState.from_dict({"task": "t", "workspace": "w", "final_response": 123})

    def test_version_mismatch_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = str(tmp_path / "s.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 999, "state": {"task": "t", "workspace": "w"}, "runtime": {}}, f)
        with pytest.raises(ValueError, match="version"):
            AgentState.load(path)

    def test_tool_call_id_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        state = AgentState(task="t", workspace="/tmp")
        state.add_message(Message(role="tool", content="out", tool_call_id="abc"))
        path = str(tmp_path / "s.json")
        state.save(path)
        loaded, _ = AgentState.load(path)
        assert loaded.messages[0].tool_call_id == "abc"


class TestSafetyRegistration:
    def test_safety_tool_registers(self) -> None:
        from minicoder.safety import SafetyTool

        reg = ToolRegistry()
        reg.register(SafetyTool(workspace="/tmp"))
        schemas = reg.list_tools()
        assert schemas[0]["name"] == "safety_check"
        assert schemas[0]["description"] != ""
        assert "check" in schemas[0]["parameter_schema"]["properties"]


class TestGitSearchRead:
    def test_git_diff_file_path(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import subprocess
        from minicoder.tools.git import GitDiffTool

        subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(tmp_path), check=True)
        (tmp_path / "a.txt").write_text("one\n")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=str(tmp_path), check=True)
        (tmp_path / "a.txt").write_text("two\n")
        out = GitDiffTool(str(tmp_path)).execute(path="a.txt")
        assert "two" in out
        assert GitDiffTool(str(tmp_path)).execute(path="missing.txt").startswith("Error:")

    def test_search_skips_huge_files(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from minicoder.tools.search import SearchFilesTool

        big = tmp_path / "big.bin"
        big.write_bytes(b"needle" + b"x" * 2_000_000)
        (tmp_path / "small.txt").write_text("nothing here\n")
        out = SearchFilesTool(str(tmp_path)).execute(pattern="needle")
        assert "skipped 1 file" in out

    def test_read_paging_and_line_cap(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from minicoder.tools.read_file import ReadFileTool

        target = tmp_path / "a.txt"
        target.write_text("".join(f"line {i}\n" for i in range(100)))
        tool = ReadFileTool(str(tmp_path))
        page = tool.execute(path="a.txt", start_line=10, end_line=12)
        assert "line 9" in page and "line 11" in page
        assert "line 8" not in page and "line 12" not in page
        capped = ReadFileTool(str(tmp_path), max_lines=5).execute(path="a.txt")
        assert "line 4" in capped
        assert "line 5" not in capped
        assert "line 50" not in capped


class TestCLIFlags:
    def test_dry_run_flag(self) -> None:
        from minicoder.cli import CLI

        cli = CLI()
        assert cli.parse_args(["task", "--dry-run"]).dry_run is True
        # Resume-missing path returns before any provider setup (no
        # network), but settings handling must already be exact.
        cli2 = CLI()
        assert cli2.run(["--resume", "--session", "/nonexistent-xyz/s.json", "--dry-run"]) == 1
        assert cli2.settings.dry_run is True

    def test_no_flag_leaves_defaults(self) -> None:
        from minicoder.cli import CLI

        cli = CLI()
        cli.run(["--session", "/nonexistent-xyz/s.json", "--resume"])
        assert cli.settings.dry_run is False


class TestTracerWiring:
    def test_run_feeds_tracer(self) -> None:
        from minicoder.tracing import Tracer

        class TwoSteps:
            def __init__(self) -> None:
                self.n = 0

            def generate(self, messages: Any, tools: Any = None) -> Any:
                self.n += 1
                if self.n == 1:
                    return {
                        "content": "go",
                        "tool_calls": [ToolCall(id="1", name="echo_tool", arguments={})],
                    }
                return {"content": "done", "tool_calls": []}

        tracer = Tracer()
        agent = Agent(
            llm_client=TwoSteps(), registry=_registry(), task="t",
            workspace="/tmp", tracer=tracer,
        )
        result = agent.run()
        assert result["finished"] is True
        traces = tracer.get_traces()
        assert len(traces) == 2
        assert len(traces[0].tool_calls) == 1
        assert traces[0].tool_calls[0].tool_name == "echo_tool"
        assert traces[0].tool_calls[0].success is True
        assert len(traces[1].tool_calls) == 0

    def test_broken_tracer_cannot_break_loop(self) -> None:
        class BadTracer:
            def start_iteration(self, *a: Any, **k: Any) -> None:
                raise RuntimeError("tracer boom")

            def __getattr__(self, name: str) -> Any:
                raise RuntimeError("tracer boom")

        class Quiet:
            def generate(self, messages: Any, tools: Any = None) -> Any:
                return {"content": "done", "tool_calls": []}

        agent = Agent(
            llm_client=Quiet(), registry=_registry(), task="t",
            workspace="/tmp", tracer=BadTracer(),
        )
        assert agent.run()["finished"] is True


class TestAskMode:
    def test_eof_denies_gracefully(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import builtins

        def _raise(*a: Any, **k: Any) -> Any:
            raise EOFError("closed stdin")

        monkeypatch.setattr(builtins, "input", _raise)
        agent = Agent(
            llm_client=None, registry=_registry(), task="t",
            workspace="/tmp", approval_mode="ask",
        )
        assert agent._check_approval("run_command", {"command": "echo hi"}) is False


class TestEmptyMessageGuards:
    def test_openai_rejects_empty(self) -> None:
        from minicoder.llm.openai import OpenAIClient

        with pytest.raises(ValueError, match="No messages"):
            OpenAIClient(model="m", api_key="k").generate([], tools=[])

    def test_anthropic_rejects_empty(self) -> None:
        from minicoder.llm.anthropic import AnthropicClient

        client = AnthropicClient.__new__(AnthropicClient)
        client._model = "m"
        with pytest.raises(ValueError, match="No messages"):
            client.generate([], tools=[])


class TestMaxHistoryConfig:
    def test_env_and_flag(self, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from minicoder.cli import CLI
        from minicoder.config import get_settings

        monkeypatch.setenv("MINICODER_MAX_HISTORY", "42")
        assert get_settings().max_history == 42
        monkeypatch.delenv("MINICODER_MAX_HISTORY")

        cli = CLI()
        parsed = cli.parse_args(["task", "--max-history", "7"])
        assert parsed.max_history == 7
        agent = Agent(
            llm_client=None, registry=_registry(), task="t",
            workspace="/tmp", max_history=parsed.max_history,
        )
        assert agent.max_history == 7

    def test_invalid_env_keeps_default(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from minicoder.config import get_settings

        monkeypatch.setenv("MINICODER_MAX_HISTORY", "0")
        assert get_settings().max_history == 100
        monkeypatch.setenv("MINICODER_MAX_HISTORY", "nope")
        assert get_settings().max_history == 100


class TestEnvFile:
    def test_custom_env_file_honored(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from minicoder.cli import CLI

        env = tmp_path / "custom.env"
        env.write_text("MINICODER_MODEL=custom-model-xyz\n", encoding="utf-8")
        monkeypatch.delenv("MINICODER_MODEL", raising=False)
        cli = CLI()
        assert cli.settings.model != "custom-model-xyz"
        # Resume-missing path returns before provider setup (no network),
        # but the env file must already have been applied.
        cli.run(["--resume", "--session", "/nonexistent-xyz/s.json",
                 "--env-file", str(env)])
        assert cli.settings.model == "custom-model-xyz"
