"""Confinement and bypass regression tests.

Covers the adversarial-review fixes: symlink escape via realpath
containment, shell-quoting blocklist bypasses, shell cwd confinement,
Error-prefix scoping to status tools, and the search aggregate budget.
"""

from __future__ import annotations

import os

from minicoder.agent import Agent
from minicoder.messages import ToolCall
from minicoder.safety import is_blocked_command
from minicoder.tools.apply_patch import ApplyPatchTool
from minicoder.tools.filesystem import ListFilesTool
from minicoder.tools.read_file import ReadFileTool
from minicoder.tools.registry import ToolRegistry
from minicoder.tools.search import SearchFilesTool
from minicoder.tools.shell import RunCommandTool
from minicoder.tools.write_file import WriteFileTool


def _outside(tmp_path_factory, name: str) -> str:  # type: ignore[no-untyped-def]
    outside = tmp_path_factory.mktemp(name)
    (outside / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")
    return str(outside)


class TestSymlinkConfinement:
    def test_read_via_symlink_blocked(self, tmp_path, tmp_path_factory) -> None:  # type: ignore[no-untyped-def]
        outside = _outside(tmp_path_factory, "outside-read")
        os.symlink(os.path.join(outside, "secret.txt"), str(tmp_path / "link.txt"))
        out = ReadFileTool(str(tmp_path)).execute(path="link.txt")
        assert "escapes workspace" in out

    def test_write_via_symlink_blocked(self, tmp_path, tmp_path_factory) -> None:  # type: ignore[no-untyped-def]
        outside = _outside(tmp_path_factory, "outside-write")
        victim = os.path.join(outside, "victim.txt")
        with open(victim, "w", encoding="utf-8") as f:
            f.write("original")
        os.symlink(victim, str(tmp_path / "wlink.txt"))
        out = WriteFileTool(str(tmp_path)).execute(path="wlink.txt", content="PWNED")
        assert "escapes workspace" in out
        with open(victim, encoding="utf-8") as f:
            assert f.read() == "original"

    def test_patch_via_symlink_blocked(self, tmp_path, tmp_path_factory) -> None:  # type: ignore[no-untyped-def]
        outside = _outside(tmp_path_factory, "outside-patch")
        os.symlink(os.path.join(outside, "secret.txt"), str(tmp_path / "plink.txt"))
        out = ApplyPatchTool(str(tmp_path)).execute(
            filepath="plink.txt", old_content="TOP", patch="X"
        )
        assert "escapes workspace" in out

    def test_inner_symlink_still_allowed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "real.txt").write_text("inner-data", encoding="utf-8")
        os.symlink(str(tmp_path / "real.txt"), str(tmp_path / "inner-link.txt"))
        assert ReadFileTool(str(tmp_path)).execute(path="inner-link.txt") == "inner-data"


class TestQuotingBypass:
    def test_backslash_escape_blocked(self) -> None:
        assert is_blocked_command("su\\do ls") is True

    def test_empty_quotes_blocked(self) -> None:
        assert is_blocked_command("s\"\"udo ls") is True
        assert is_blocked_command("s''udo ls") is True

    def test_quoted_blocked_word_still_blocked(self) -> None:
        assert is_blocked_command('"sudo" ls') is True

    def test_true_negatives_still_allowed(self) -> None:
        assert is_blocked_command("echo address") is False
        assert is_blocked_command("echo hello") is False
        assert is_blocked_command("cat mysudoers") is False


class TestCwdConfinement:
    def test_cwd_outside_falls_back(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        tool = RunCommandTool(workspace=str(tmp_path))
        result = tool.execute(command="pwd", cwd="..")
        assert result["success"] is True
        assert result["stdout"].strip() == os.path.realpath(str(tmp_path))

    def test_cwd_inside_still_honored(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "sub").mkdir()
        tool = RunCommandTool(workspace=str(tmp_path))
        result = tool.execute(command="pwd", cwd="sub")
        assert result["stdout"].strip() == os.path.realpath(str(tmp_path / "sub"))


class TestErrorPrefixScoping:
    def test_error_like_content_not_flagged(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "err.txt").write_text(
            "Error: something bad happened\nmore\n", encoding="utf-8"
        )
        reg = ToolRegistry()
        reg.register(ReadFileTool(workspace=str(tmp_path)))
        agent = Agent(llm_client=None, registry=reg, task="t", workspace=str(tmp_path))
        agent._execute_tool_call(
            ToolCall(id="1", name="read_file", arguments={"path": "err.txt"})
        )
        assert agent.state.errors == []

    def test_mutating_tool_errors_still_recorded(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        reg = ToolRegistry()
        reg.register(WriteFileTool(workspace=str(tmp_path)))
        agent = Agent(llm_client=None, registry=reg, task="t", workspace=str(tmp_path))
        agent._execute_tool_call(
            ToolCall(id="1", name="write_file", arguments={"path": "../x", "content": "y"})
        )
        assert any("failed" in e for e in agent.state.errors)


class TestSearchBudget:
    def test_total_budget_note_deterministic(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("needle " + "x" * 200 + "\n", encoding="utf-8")
        tool = SearchFilesTool(str(tmp_path), max_total_bytes=100)
        first = tool.execute(pattern="needle")
        second = tool.execute(pattern="needle")
        assert first == second
        assert "size budget" in first

    def test_small_search_unaffected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "a.txt").write_text("needle here\n", encoding="utf-8")
        out = SearchFilesTool(str(tmp_path)).execute(pattern="needle")
        assert "Line 1" in out
        assert "budget" not in out


class TestListEscape:
    def test_list_outside_blocked(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        out = ListFilesTool(str(tmp_path)).execute(path="../..")
        assert "escapes workspace" in out
