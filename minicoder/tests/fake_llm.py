"""Fake LLM client for testing the MiniCoder agent loop.

This simulates a genuine coding-agent trajectory:
1. list_files
2. read_file
3. search_files
4. apply_patch
5. run_command
6. observe failure
7. read_file
8. apply_patch
9. run_command
10. observe success
11. finish
"""

from __future__ import annotations
from typing import Any, List, Optional, Dict
from minicoder.messages import Message, ToolCall


class FakeLLMClient:
    """Fake LLM that simulates a coding agent trajectory.

    The trajectory simulates a real agent finding and fixing bugs
    in the demo project.
    """

    def __init__(self, task: str = "") -> None:
        self.task = task
        self._step: int = 0
        # Track which tools have been called
        self.call_history: List[dict[str, Any]] = []

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Generate a response following the coding agent trajectory.

        Step 0: list_files - explore the workspace
        Step 1: read_file - examine relevant files
        Step 2: search_files - search for patterns
        Step 3: apply_patch - fix the first defect
        Step 4: run_command - run tests (fail)
        Step 5: observe failure
        Step 6: read_file - re-examine
        Step 7: apply_patch - fix the second defect
        Step 8: run_command - run tests (pass)
        Step 9: observe success
        Step 10: finish
        """
        # Get the tool schemas from registry (passed by agent)
        # But since we're fake, we just use the tool names

        # Determine response based on current step
        tool_calls: List[Any] = []
        content: Optional[str] = None
        step = self._step

        if step == 0:
            # First step: list files
            content = "I'll start by exploring the repository structure to understand the project layout."
            tool_calls.append(self._make_tool_call("list_files", {"path": "."}))

        elif step == 1:
            # Read relevant files
            content = "Now I'll read the calculator and validator files to understand the code and find the bugs."
            tool_calls.append(self._make_tool_call("read_file", {"path": "app/calculator.py"}))
            tool_calls.append(self._make_tool_call("read_file", {"path": "app/validators.py"}))
            tool_calls.append(self._make_tool_call("read_file", {"path": "tests/test_calculator.py"}))
            tool_calls.append(self._make_tool_call("read_file", {"path": "tests/test_validators.py"}))

        elif step == 2:
            # Search for patterns
            content = "I'll search for the specific patterns that reveal the bugs."
            tool_calls.append(self._make_tool_call("search_files", {"pattern": "add|subtract|validate"}))
            # Actually let's do search after reading

        elif step == 3:
            # Apply patch for first defect
            content = "I've identified the bugs. Now I'll fix the first defect in calculator.py."
            # Fix: change add() from a-b to a+b
            tool_calls.append(self._make_tool_call("apply_patch", {
                "filepath": "app/calc_test.py",
                "old_content": 'def add(a: int, b: int) -> int:\n    """Add two numbers. BUG: subtracts instead of adding."""\n    return a - b',
                "patch": 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b'
            }))

        elif step == 4:
            # Run tests - they should fail now (first defect partially fixed but not all)
            content = "Running tests to verify the fix..."
            tool_calls.append(self._make_tool_call("run_command", {
                "command": "cd /home/shivam/minicoder && python -m pytest demo_project/tests/test_calculator.py -v",
                "cwd": "/home/shivam/minicoder"
            }))

        elif step == 5:
            # Observe failure - tests still fail because second defect exists
            content = "Tests failed as expected - there's still a second defect to fix in validators.py."
            tool_calls.append(self._make_tool_call("apply_patch", {
                "filepath": "app/val_test.py",
                "old_content": 'def validate_password(password: str) -> bool:\n    """Validate password. BUG: accepts short passwords."""\n    if len(password) < 3:  # BUG: should be < 8\n        return False',
                "patch": 'def validate_password(password: str) -> bool:\n    """Validate password."""\n    if len(password) < 8:\n        return False'
            }))

        elif step == 6:
            # Read validators again
            content = "Now I'll read the validators file again to confirm the fix."
            tool_calls.append(self._make_tool_call("read_file", {"path": "app/validators.py"}))

        elif step == 7:
            # Apply patch for second defect (already done in step 5, just continuing)
            content = "Fixing the second defect in validators.py."
            # This step might re-apply or just continue

        elif step == 8:
            # Run tests again - should pass now
            content = "Running tests to verify both fixes..."
            tool_calls.append(self._make_tool_call("run_command", {
                "command": "cd /home/shivam/minicoder && python -m pytest demo_project/tests/ -v",
                "cwd": "/home/shivam/minicoder"
            }))

        elif step == 9:
            # Observe success
            content = "All tests pass! The agent has successfully fixed all defects."
            tool_calls.append(self._make_tool_call("finish", {
                "summary": "Fixed two intentional defects in calculator.py and validators.py. "
                          "Calculator functions now correctly add and subtract. "
                          "Validators now properly check email format, password length (>=8), "
                          "and username alphanumeric constraints.",
                "tests_run": 6,
                "remaining_issues": 0,
            }))

        else:
            # Beyond step 9, finish
            content = "Task complete. All defects have been fixed and verified."
            self._step = 10  # Ensure we don't loop

        self._step += 1

        # Build response
        result: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
            "tool_calls": tool_calls,
        }

        # Store in call history
        self.call_history.append({
            "step": step - 1,  # The step we just responded from
            "tool_calls": [tc.model_dump() if hasattr(tc, 'model_dump') else tc for tc in tool_calls],
            "content": content,
        })

        return result

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Streaming is not supported by the fake client."""
        return self.generate(*args, **kwargs)

    def _make_tool_call(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        """Create a ToolCall for the fake model."""
        import uuid
        return ToolCall(
            id=str(uuid.uuid4()),
            name=name,
            arguments=arguments,
        )