"""Tests for context-budget truncation across tools."""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import patch

from minicoder.tools.budget import (
    DEFAULT_MAX_FILE_CHARS,
    DEFAULT_MAX_OUTPUT_CHARS,
    truncate_text,
)
from minicoder.tools.filesystem import ListFilesTool
from minicoder.tools.git import GitDiffTool, GitStatusTool
from minicoder.tools.read_file import ReadFileTool
from minicoder.tools.search import SearchFilesTool
from minicoder.tools.shell import RunCommandTool


def _write(path, content: str) -> None:  # type: ignore[no-untyped-def]
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class TestTruncateHelper:
    def test_under_limit_unchanged(self) -> None:
        assert truncate_text("abc", 10) == ("abc", False)

    def test_exact_boundary_not_truncated(self) -> None:
        text = "x" * 100
        out, truncated = truncate_text(text, 100)
        assert truncated is False
        assert out == text

    def test_one_over_limit_truncated(self) -> None:
        out, truncated = truncate_text("x" * 101, 100)
        assert truncated is True
        assert out.startswith("x" * 100)
        assert "truncated 1 chars" in out

    def test_deterministic(self) -> None:
        text = "abcdefghij"
        assert truncate_text(text, 4) == truncate_text(text, 4)

    def test_hint_included(self) -> None:
        out, truncated = truncate_text("abcdef", 2, hint="(do y)")
        assert truncated is True
        assert "(do y)" in out


class TestReadFileBudget:
    def test_small_file_complete(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        _write(tmp_path / "a.txt", "hello\nworld\n")
        out = ReadFileTool(str(tmp_path)).execute(path="a.txt")
        assert out == "hello\nworld\n"

    def test_large_file_truncated_with_hint(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        _write(tmp_path / "big.txt", "y\n" * 20000)
        tool = ReadFileTool(str(tmp_path), max_chars=100)
        out = tool.execute(path="big.txt")
        assert len(out) < 20000 * 2
        assert "truncated" in out
        assert "start_line" in out  # narrowing hint preserved

    def test_boundary_exact(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        _write(tmp_path / "a.txt", "12345")
        assert "truncated" not in ReadFileTool(str(tmp_path), max_chars=5).execute(path="a.txt")
        out = ReadFileTool(str(tmp_path), max_chars=4).execute(path="a.txt")
        assert "truncated" in out
        assert out.startswith("1234")

    def test_configurable(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        _write(tmp_path / "a.txt", "z" * 50)
        assert "truncated" not in ReadFileTool(str(tmp_path), max_chars=1000).execute(path="a.txt")


class TestSearchBudget:
    def _ws(self, tmp_path, n_files: int, matches_per_file: int = 1):  # type: ignore[no-untyped-def]
        for i in range(n_files):
            lines = "".join(f"needle line {j}\n" for j in range(matches_per_file))
            _write(tmp_path / f"f{i:03d}.py", lines + "other\n")
        return str(tmp_path)

    def test_file_cap_with_omitted_note(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        ws = self._ws(tmp_path, 60)
        out = SearchFilesTool(ws, max_files=50).execute(pattern="needle")
        assert "more files" in out and "omitted" in out
        assert out.count("File:") == 50

    def test_match_cap_with_omitted_note(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        ws = self._ws(tmp_path, 1, matches_per_file=10)
        out = SearchFilesTool(ws, max_matches_per_file=5).execute(pattern="needle")
        assert "more matches" in out and "omitted" in out

    def test_long_line_truncated(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        _write(tmp_path / "a.py", "needle " + "z" * 500 + "\n")
        out = SearchFilesTool(str(tmp_path), max_line_chars=20).execute(pattern="needle")
        assert "…" in out

    def test_deterministic_order(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        ws = self._ws(tmp_path, 5)
        first = SearchFilesTool(ws).execute(pattern="needle")
        second = SearchFilesTool(ws).execute(pattern="needle")
        assert first == second
        files = [l for l in first.splitlines() if l.startswith("File:")]
        assert files == sorted(files)

    def test_total_char_cap(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        ws = self._ws(tmp_path, 10, matches_per_file=20)
        out = SearchFilesTool(str(tmp_path), max_chars=200).execute(pattern="needle")
        assert "truncated" in out


class TestListFilesBudget:
    def test_entry_cap_with_note(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        for i in range(30):
            _write(tmp_path / f"f{i:02d}.txt", "x")
        out = ListFilesTool(str(tmp_path), max_files=10).execute()
        assert "more files" in out and "omitted" in out
        assert len([l for l in out.splitlines() if l.endswith(".txt")]) == 10

    def test_small_listing_complete(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        _write(tmp_path / "a.txt", "x")
        out = ListFilesTool(str(tmp_path)).execute()
        assert out.strip() == "a.txt"


class TestShellBudget:
    def test_boundary_flags(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        tool = RunCommandTool(str(tmp_path), max_output_chars=10)
        ok = tool.execute(command="echo hi")
        assert ok["truncated"] is False
        big = tool.execute(command="python3 -c \"print('x' * 100)\"")
        assert big["truncated"] is True
        assert "truncated" in big["stdout"]

    def test_configurable_limit(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        tool = RunCommandTool(str(tmp_path), max_output_chars=100000)
        result = tool.execute(command="python3 -c \"print('x' * 5000)\"")
        assert result["truncated"] is False
        assert "xxxxx" in result["stdout"]


class TestGitBudget:
    def _fake(self, text: str):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout=text, stderr=""
        )

    def test_diff_truncated(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with patch("subprocess.run", return_value=self._fake("d" * 500)):
            out = GitDiffTool(str(tmp_path), max_chars=100).execute()
        assert "truncated" in out
        assert out.startswith("d" * 100)

    def test_status_truncated(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with patch("subprocess.run", return_value=self._fake("s" * 500)):
            out = GitStatusTool(str(tmp_path), max_chars=100).execute(porcelain=True)
        assert "truncated" in out

    def test_small_output_untouched(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with patch("subprocess.run", return_value=self._fake("ok")):
            assert GitDiffTool(str(tmp_path)).execute() == "ok"


class TestBudgetConfig:
    def test_env_override(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from minicoder.config import get_settings

        monkeypatch.setenv("MINICODER_MAX_TOOL_OUTPUT_CHARS", "1234")
        assert get_settings().max_tool_output_chars == 1234

    def test_invalid_env_ignored(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from minicoder.config import get_settings

        monkeypatch.setenv("MINICODER_MAX_TOOL_OUTPUT_CHARS", "not-a-number")
        assert (
            get_settings().max_tool_output_chars == DEFAULT_MAX_OUTPUT_CHARS
        )

    def test_defaults_documented(self) -> None:
        assert DEFAULT_MAX_OUTPUT_CHARS == 8000
        assert DEFAULT_MAX_FILE_CHARS == 12000
