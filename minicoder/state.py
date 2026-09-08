from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any
from .messages import Message, ToolCall, ToolResult


@dataclass
class AgentState:
    task: str
    workspace: str
    messages: List[Message] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 100
    tool_calls_executed: List[ToolCall] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    test_results: Optional[dict[str, bool]] = None
    errors: List[str] = field(default_factory=list)
    finished: bool = False
    final_response: Optional[str] = None

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def can_proceed(self) -> bool:
        return not self.finished and self.current_iteration < self.max_iterations

    def summary(self) -> str:
        lines = [
            f"Task: {self.task}",
            f"Iteration: {self.current_iteration}/{self.max_iterations}",
            f"Finished: {self.finished}",
            f"Errors: {len(self.errors)}",
            f"Modified files: {len(self.modified_files)}",
        ]
        return "\n".join(lines)