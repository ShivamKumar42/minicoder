from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from .messages import Message, ToolCall


SESSION_VERSION = 1
"""Version of the on-disk session envelope format."""


def _tool_call_to_dict(tc: ToolCall) -> Dict[str, Any]:
    return {"id": tc.id, "name": tc.name, "arguments": tc.arguments}


def _tool_call_from_dict(data: Any) -> ToolCall:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid tool call entry: {data!r}")
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"Tool call arguments must be an object: {data!r}")
    return ToolCall(
        id=str(data.get("id", "")),
        name=str(data.get("name", "")),
        arguments=arguments,
    )


def _message_to_dict(msg: Message) -> Dict[str, Any]:
    return {
        "role": msg.role,
        "content": msg.content,
        "tool_calls": (
            [_tool_call_to_dict(tc) for tc in msg.tool_calls]
            if msg.tool_calls
            else None
        ),
        "tool_call_id": msg.tool_call_id,
    }


def _message_from_dict(data: Any) -> Message:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid message entry: {data!r}")
    raw_calls = data.get("tool_calls")
    tool_calls = (
        [_tool_call_from_dict(tc) for tc in raw_calls]
        if raw_calls
        else None
    )
    call_id = data.get("tool_call_id")
    if call_id is not None and not isinstance(call_id, str):
        raise ValueError(f"Invalid tool_call_id: {call_id!r}")
    return Message(
        role=str(data.get("role", "user")),
        content=str(data.get("content", "")),
        tool_calls=tool_calls,
        tool_call_id=call_id,
    )


def _to_int(data: Any, field_name: str, default: int) -> int:
    """Coerce session ints, raising ValueError (not TypeError) on garbage.

    An explicit null/wrong-typed value is corruption and is rejected;
    callers pass the default when the key is absent.
    """
    if isinstance(data, bool) or not isinstance(data, int):
        raise ValueError(f"Invalid {field_name}: {data!r}")
    return data


def _to_str_list(data: Any, field_name: str) -> List[str]:
    if not isinstance(data, list):
        raise ValueError(f"Invalid {field_name}: {data!r}")
    return [str(e) for e in data]


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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to plain JSON-compatible types (no secrets)."""
        return {
            "task": self.task,
            "workspace": self.workspace,
            "messages": [_message_to_dict(m) for m in self.messages],
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "tool_calls_executed": [
                _tool_call_to_dict(tc) for tc in self.tool_calls_executed
            ],
            "modified_files": list(self.modified_files),
            "test_results": (
                dict(self.test_results) if self.test_results is not None else None
            ),
            "errors": list(self.errors),
            "finished": self.finished,
            "final_response": self.final_response,
        }

    @classmethod
    def from_dict(cls, data: Any) -> AgentState:
        """Restore state from `to_dict()` output. Raises ValueError."""
        if not isinstance(data, dict):
            raise ValueError(f"Invalid session state: {data!r}")
        test_results = data.get("test_results")
        if test_results is not None and not isinstance(test_results, dict):
            raise ValueError(f"Invalid test_results: {test_results!r}")
        raw_messages = data.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError(f"Invalid messages: {raw_messages!r}")
        raw_executed = data.get("tool_calls_executed", [])
        if not isinstance(raw_executed, list):
            raise ValueError(
                f"Invalid tool_calls_executed: {raw_executed!r}"
            )
        final_response = data.get("final_response")
        if final_response is not None and not isinstance(final_response, str):
            raise ValueError(f"Invalid final_response: {final_response!r}")
        return cls(
            task=str(data.get("task", "")),
            workspace=str(data.get("workspace", "")),
            messages=[_message_from_dict(m) for m in raw_messages],
            current_iteration=_to_int(
                data.get("current_iteration", 0), "current_iteration", 0
            ),
            max_iterations=_to_int(
                data.get("max_iterations", 100), "max_iterations", 100
            ),
            tool_calls_executed=[
                _tool_call_from_dict(tc) for tc in raw_executed
            ],
            modified_files=_to_str_list(data.get("modified_files", []), "modified_files"),
            test_results=(
                {str(k): bool(v) for k, v in test_results.items()}
                if test_results is not None
                else None
            ),
            errors=_to_str_list(data.get("errors", []), "errors"),
            finished=bool(data.get("finished", False)),
            final_response=final_response,
        )

    def save(self, path: str, total_tool_calls: int = 0) -> str:
        """Write a session envelope to disk (JSON, no secrets)."""
        envelope = {
            "version": SESSION_VERSION,
            "state": self.to_dict(),
            "runtime": {"total_tool_calls": total_tool_calls},
        }
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> Tuple[AgentState, int]:
        """Load a session envelope. Returns (state, total_tool_calls).

        Raises FileNotFoundError for missing files and ValueError for
        corrupted/invalid content.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Session file not found: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                envelope = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted session file {path}: {e}") from e
        except OSError as e:
            raise ValueError(f"Cannot read session file {path}: {e}") from e
        if not isinstance(envelope, dict) or "state" not in envelope:
            raise ValueError(f"Invalid session file {path}: missing 'state'")
        version = envelope.get("version", SESSION_VERSION)
        if version != SESSION_VERSION:
            raise ValueError(
                f"Unsupported session version {version!r} in {path} "
                f"(supports {SESSION_VERSION})"
            )
        runtime = envelope.get("runtime", {})
        total = 0
        if isinstance(runtime, dict):
            raw_total = runtime.get("total_tool_calls", 0)
            if isinstance(raw_total, bool) or not isinstance(raw_total, int):
                raise ValueError(
                    f"Invalid session file {path}: bad total_tool_calls"
                )
            total = raw_total
        elif runtime:
            raise ValueError(
                f"Invalid session file {path}: bad runtime section"
            )
        return cls.from_dict(envelope["state"]), total