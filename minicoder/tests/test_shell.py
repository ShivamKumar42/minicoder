"""Regression tests for RunCommandTool shell fixes.

Covers: normal command, pipe, && / cd, non-zero exit (structured,
no crash), large-output truncation, blocked unsafe command.
"""

from __future__ import annotations

from minicoder.tools.shell import RunCommandTool


def _tool(tmp_path) -> RunCommandTool:  # type: ignore[no-untyped-def]
    return RunCommandTool(workspace=str(tmp_path), timeout=30)


class TestShellCommands:
    def test_normal_command(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _tool(tmp_path).execute(command="echo hello")
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    def test_pipe(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _tool(tmp_path).execute(command="echo hello | tr a-z A-Z")
        assert result["success"] is True
        assert "HELLO" in result["stdout"]

    def test_and_chain_and_cd(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        sub = tmp_path / "sub"
        sub.mkdir()
        result = _tool(tmp_path).execute(command="cd sub && pwd && echo ok")
        assert result["success"] is True
        assert "ok" in result["stdout"]
        assert "sub" in result["stdout"]

    def test_nonzero_exit_is_structured(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _tool(tmp_path).execute(command="ls /nonexistent-minicoder-dir-xyz")
        assert result["success"] is False
        assert result["exit_code"] != 0
        # stderr preserved for the agent; loop must not crash.
        assert result["stderr"] != ""
        assert set(result.keys()) >= {"stdout", "stderr", "exit_code", "success"}

    def test_large_output_truncated(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        tool = _tool(tmp_path)
        result = tool.execute(
            command="python3 -c \"print('x' * 50000)\""
        )
        assert result["success"] is True
        assert len(result["stdout"]) < 50000
        assert len(result["stdout"]) <= tool.MAX_OUTPUT_CHARS + 100
        assert "truncated" in result["stdout"].lower()
        assert result.get("truncated") is True

    def test_blocked_unsafe_command(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _tool(tmp_path).execute(command="sudo ls /")
        assert result["success"] is False
        assert result["exit_code"] == 1
        assert "blocked" in result["stderr"].lower()

    def test_blocked_pipe_rm(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _tool(tmp_path).execute(command="echo hi | rm foo")
        assert result["success"] is False
        assert "blocked" in result["stderr"].lower()

    def test_unknown_cwd_falls_back_to_workspace(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        result = _tool(tmp_path).execute(command="echo hi", cwd="does-not-exist")
        assert result["success"] is True
        assert "hi" in result["stdout"]
