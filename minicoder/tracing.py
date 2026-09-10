from __future__ import annotations
import os
import time
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ToolTrace:
    """Trace information for a single tool execution."""
    tool_name: str
    arguments: dict[str, Any]
    start_time: float
    duration: float
    success: bool
    exit_code: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "start_time": self.start_time,
            "duration": self.duration,
            "success": self.success,
            "exit_code": self.exit_code,
            "error": self.error,
        }


@dataclass
class AgentTrace:
    """Trace information for the entire agent run."""
    iteration: int
    model: str
    provider: str
    tool_calls: List[ToolTrace] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    final_state: Optional[dict[str, Any]] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "model": self.model,
            "provider": self.provider,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "failures": self.failures,
            "modified_files": self.modified_files,
            "final_state": self.final_state,
        }


class Tracer:
    """Structured tracing for agent operations."""
    
    def __init__(self, verbose: bool = False, save_path: Optional[str] = None):
        self.verbose = verbose
        self.save_path = save_path
        self._agent_traces: List[AgentTrace] = []
        self._tool_traces: List[ToolTrace] = []
        self._current_iteration = 0
        self._current_model = ""
        self._current_provider = ""
    
    def start_iteration(self, iteration: int, model: str, provider: str) -> None:
        """Start tracing a new iteration."""
        self._current_iteration = iteration
        self._current_model = model
        self._current_provider = provider
        self._tool_traces = []
    
    def start_tool(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Start tracing a tool execution."""
        self._tool_traces.append(ToolTrace(
            tool_name=tool_name,
            arguments=arguments,
            start_time=time.time(),
            duration=0.0,
            success=False,
        ))
    
    def end_tool(self, success: bool, exit_code: Optional[int] = None, error: Optional[str] = None) -> None:
        """End tracing a tool execution."""
        if self._tool_traces:
            last = self._tool_traces[-1]
            last.duration = time.time() - last.start_time
            last.success = success
            last.exit_code = exit_code
            last.error = error
    
    def record_agent_iteration(self) -> AgentTrace:
        """Record the current iteration's agent trace."""
        trace = AgentTrace(
            iteration=self._current_iteration,
            model=self._current_model,
            provider=self._current_provider,
            tool_calls=self._tool_traces.copy(),
        )
        self._agent_traces.append(trace)
        return trace
    
    def get_traces(self) -> List[AgentTrace]:
        """Get all recorded agent traces."""
        return self._agent_traces
    
    def save_to_json(self, path: str) -> None:
        """Save all traces to a JSON file (creating parent dirs)."""
        data = [trace.to_dict() for trace in self._agent_traces]
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[TRACE] {message}")
    
    def log_tool(self, tool_name: str, args: dict[str, Any], duration: float, success: bool) -> None:
        """Log tool execution if verbose mode is enabled."""
        if self.verbose:
            status = "OK" if success else "FAIL"
            print(f"[TRACE] Tool: {tool_name}, Duration: {duration:.3f}s, Status: {status}, Args: {str(args)[:100]}")
    
    def log_agent(self, iteration: int, model: str, tool_calls_count: int, failures: int) -> None:
        """Log agent iteration if verbose mode is enabled."""
        if self.verbose:
            print(f"[TRACE] Iteration: {iteration}, Model: {model}, Tool calls: {tool_calls_count}, Failures: {failures}")