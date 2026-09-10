"""Regression tests for ApplyPatchTool minimal replacement semantics.

Failure modes previously identified in the audit:
- unified-diff parser crashed (NameError) / always returned None
- direct-replacement fallback overwrote the entire file silently
- `old_content` anchor was ignored
- empty anchor / ambiguous matches / missing files not handled safely
"""

from __future__ import annotations

from minicoder.tools.apply_patch import ApplyPatchTool


def _write(path, content: str) -> None:  # type: ignore[no-untyped-def]
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _read(path) -> str:  # type: ignore[no-untyped-def]
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestApplyPatch:
    def test_valid_multiline_replacement(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        target = tmp_path / "calc.py"
        _write(target, 'def add(a, b):\n    """BUG."""\n    return a - b\n')
        tool = ApplyPatchTool(workspace=str(tmp_path))
        result = tool.execute(
            filepath="calc.py",
            old_content="    return a - b",
            patch="    return a + b",
        )
        assert "Successfully applied patch" in result
        assert "return a + b" in _read(target)
        assert "return a - b" not in _read(target)

    def test_fake_llm_style_call(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Exact shape used by tests/fake_llm.py (old snippet -> new text)."""
        target = tmp_path / "calc.py"
        _write(
            target,
            'def add(a: int, b: int) -> int:\n    """Add two numbers. BUG."""\n    return a - b\n',
        )
        tool = ApplyPatchTool(workspace=str(tmp_path))
        result = tool.execute(
            filepath="calc.py",
            old_content='def add(a: int, b: int) -> int:\n    """Add two numbers. BUG."""\n    return a - b',
            patch='def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b',
        )
        assert "Successfully" in result
        content = _read(target)
        assert "return a + b" in content
        assert "BUG" not in content

    def test_rest_of_file_preserved(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Replacement must not silently overwrite the entire file."""
        original = "line1\nline2 BUG\nline3\nline4\n"
        target = tmp_path / "a.txt"
        _write(target, original)
        tool = ApplyPatchTool(workspace=str(tmp_path))
        result = tool.execute(filepath="a.txt", old_content="line2 BUG", patch="line2 FIXED")
        assert "Successfully" in result
        assert _read(target) == "line1\nline2 FIXED\nline3\nline4\n"

    def test_old_content_not_found_fails_safely(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = "line1\nline2\n"
        target = tmp_path / "a.txt"
        _write(target, original)
        tool = ApplyPatchTool(workspace=str(tmp_path))
        result = tool.execute(filepath="a.txt", old_content="no such snippet", patch="x")
        assert "Error" in result
        assert "not found" in result.lower()
        assert _read(target) == original  # unchanged

    def test_nonexistent_target_reported(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        tool = ApplyPatchTool(workspace=str(tmp_path))
        result = tool.execute(filepath="missing.py", old_content="x", patch="y")
        assert "Error" in result
        assert "not found" in result.lower() or "File not found" in result

    def test_empty_old_content_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = "abc"
        target = tmp_path / "a.txt"
        _write(target, original)
        tool = ApplyPatchTool(workspace=str(tmp_path))
        result = tool.execute(filepath="a.txt", old_content="", patch="OVERWRITE")
        assert "Error" in result
        assert _read(target) == original  # must not touch the file

    def test_ambiguous_match_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        original = "x = 1\nx = 1\n"
        target = tmp_path / "a.txt"
        _write(target, original)
        tool = ApplyPatchTool(workspace=str(tmp_path))
        result = tool.execute(filepath="a.txt", old_content="x = 1", patch="x = 2")
        assert "Error" in result
        assert _read(target) == original  # unchanged on ambiguity

    def test_unified_diff_input_fails_safely(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Previously crashed with NameError; must now fail as a safe error."""
        original = "line1\nline2\nline3\n"
        target = tmp_path / "a.txt"
        _write(target, original)
        tool = ApplyPatchTool(workspace=str(tmp_path))
        result = tool.execute(
            filepath="a.txt",
            old_content="line2",
            patch="@@ -2,1 +2,1 @@\n-line2\n+LINE2",
        )
        # Either applied literally (anchor found) or, more likely here,
        # treated as literal replacement text -- but must never crash and
        # must never destroy the rest of the file.
        assert isinstance(result, str)
        assert "line1" in _read(target) and "line3" in _read(target)

    def test_path_escape_blocked(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        tool = ApplyPatchTool(workspace=str(tmp_path))
        result = tool.execute(filepath="../escape.txt", old_content="x", patch="y")
        assert "Error" in result
        assert "escapes workspace" in result
