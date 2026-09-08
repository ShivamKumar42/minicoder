from __future__ import annotations
import json
import sys
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .messages import Message, ToolCall, ToolResult
from .tools.registry import ToolRegistry
from .state import AgentState


class Agent:
    """Main agent orchestration loop."""
    
    def __init__(
        self,
        llm_client: Any,
        registry: ToolRegistry,
        task: str,
        workspace: str,
        max_iterations: int = 100,
        max_tool_calls: int = 200,
        approval_mode: str = "auto",
        dry_run: bool = False,
        tracer: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.registry = registry
        self.task = task
        self.workspace = workspace
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.approval_mode = approval_mode
        self.dry_run = dry_run
        self.tracer = tracer
        self.verbose = verbose
        
        # Initialize state
        self.state = AgentState(
            task=task,
            workspace=workspace,
        )
        
        # Tool call tracking
        self.total_tool_calls = 0
        self.tool_call_history: List[dict[str, Any]] = []
    
    def run(self) -> dict[str, Any]:
        """Run the agent loop until completion."""
        iteration = 0
        
        while self.state.can_proceed():
            iteration += 1
            self.state.current_iteration = iteration
            
            if self.verbose:
                print(f"[AGENT] Iteration {iteration}")
            
            # Generate response from LLM
            try:
                response = self._generate_response()
            except Exception as e:
                self.state.add_error(f"LLM generation error: {str(e)}")
                if self.verbose:
                    print(f"[AGENT] LLM generation error: {e}")
                iteration -= 1
                continue
            
            # Process response
            self._process_response(response)
            
            # Check if we should continue
            if self.state.finished:
                break
            
            # Check tool call limits
            if self.total_tool_calls >= self.max_tool_calls:
                self.state.add_error(
                    f"Maximum tool calls ({self.max_tool_calls}) reached"
                )
                break
        
        # Final state
        self.state.finished = True
        
        result: dict[str, Any] = {
            "finished": self.state.finished,
            "final_response": self.state.final_response,
            "modified_files": [
                f for f in self.state.modified_files
            ],
            "errors": list(self.state.errors),
            "iterations": iteration,
            "tool_calls": self.total_tool_calls,
        }
        
        return result
    
    def _generate_response(self) -> Any:
        """Generate a response from the LLM."""
        messages = self.state.messages
        
        # Add system prompt with task context
        system_msg = Message(
            role="system",
            content=SYSTEM_PROMPT,
        )
        
        # Build the message list with system prompt
        all_messages = [system_msg] + messages
        
        # Convert to provider format
        tools = self.registry.get_tool_schemas()
        
        response = self.llm_client.generate(
            messages=all_messages,
            tools=tools,
        )
        
        return response
    
    def _process_response(self, response: Any) -> None:
        """Process the LLM response and execute tool calls if present."""
        tool_calls = response.get('tool_calls') if isinstance(response, dict) else getattr(response, 'tool_calls', None)
        content = getattr(response, 'content', None)
        
        if tool_calls:
            # Execute tool calls
            for tool_call in tool_calls:
                self._execute_tool_call(tool_call)
            
            # Append tool results to messages
            self._append_tool_results(tool_calls)
            
        elif content:
            # Model produced a final response without tool calls
            self.state.final_response = content
            self.state.finished = True
            
            if self.verbose:
                print(f"[AGENT] Final response: {content[:100]}...")
    
    def _execute_tool_call(self, tool_call: ToolCall) -> None:
        """Execute a single tool call."""
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        
        self.total_tool_calls += 1
        
        # Look up the tool
        tool = self.registry.lookup(tool_name)
        if tool is None:
            error_msg = f"Unknown tool: {tool_name}"
            self.state.add_error(error_msg)
            self._append_message(
                Message(
                    role="tool",
                    content=error_msg,
                    tool_calls=[tool_call],
                )
            )
            return
        
        # Check approval mode
        if not self._check_approval(tool_name, tool_args):
            return
        
        # Execute the tool
        try:
            result = tool.execute(**tool_args)
            
            # Success path
            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                output=result if isinstance(result, str) else str(result),
                success=True,
            )
            
            self._append_message(
                Message(
                    role="tool",
                    content=tool_result.output,
                )
            )
            
            # Check if this is a finish tool
            if tool_name == "finish":
                self.state.finished = True
                self.state.final_response = result if isinstance(result, str) else str(result)
            
        except Exception as e:
            # Error path
            error_msg = f"Tool execution error: {str(e)}"
            self.state.add_error(error_msg)
            
            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                output="",
                success=False,
                metadata={"error": str(e)},
            )
            
            self._append_message(
                Message(
                    role="tool",
                    content=error_msg,
                )
            )
    
    def _check_approval(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Check approval mode before executing a tool."""
        if self.approval_mode == "auto":
            return True
        elif self.approval_mode == "deny-dangerous":
            # Block obviously dangerous tools
            dangerous_tools = {"run_command"}
            if tool_name in dangerous_tools:
                # Try to check for destructive commands
                cmd = tool_args.get("command", "")
                if any(
                    kw in cmd.lower()
                    for kw in ["rm -rf", "sudo", "shutdown", "format", "dd"]
                ):
                    print(
                        f"[APPROVAL] Denied dangerous command: {cmd[:80]}...")
                    return False
            return True
        elif self.approval_mode == "ask":
            print(f"[APPROVAL] Approve tool {tool_name} with args {str(tool_args)[:100]}?")
            response = input("[y/n/q] ").strip().lower()
            if response == 'y':
                return True
            elif response == 'q':
                print("[APPROVAL] Quitting at user request")
                self.state.finished = True
                return False
            else:
                print("[APPROVAL] Rejected by user")
                return False
        else:
            return True
    
    def _append_tool_results(self, tool_calls: List[ToolCall]) -> None:
        """Append tool results to messages (called after execution)."""
        # Tool results are appended in _execute_tool_call
        pass
    
    def _append_message(self, message: Message) -> None:
        """Append a message to the state."""
        self.state.add_message(message)


# System prompt constant
SYSTEM_PROMPT = """You are MiniCoder, an autonomous coding agent. Your task is to help users complete software engineering tasks through iterative tool use and verification.

CORE INSTRUCTIONS:
1. INSPECT BEFORE MODIFYING - Always examine existing code, tests, and project structure before making changes. Understand the project conventions and existing patterns.
2. SEARCH BEFORE GUESSING - Use the search_files tool to find relevant code, patterns, and definitions rather than guessing or memorizing.
3. READ RELEVANT CODE - Use read_file to examine files you plan to modify. Understand the full context before applying patches.
4. UNDERSTAND EXISTING PROJECT CONVENTIONS - Follow the existing code style, naming conventions, and project patterns. Do not impose arbitrary changes.
5. MAKE MINIMAL CHANGES - Only change what is necessary to fix the problem or complete the task. Avoid unrelated modifications.
6. PREFER apply_patch FOR EDITS - Use the apply_patch tool for targeted changes. This is the preferred method for modifying existing code.
7. RUN TESTS AFTER MEANINGFUL MODIFICATIONS - After any code change, run the relevant tests to verify correctness. Treat command output as ground truth.
8. TREAT COMMAND OUTPUT AS GROUND TRUTH - If a command fails or produces unexpected output, investigate and understand why before proceeding.
9. INVESTIGATE FAILURES - When tests fail or commands error, analyze the root cause before making repairs. Do not claim success without verification.
10. ITERATE WHEN TESTS FAIL - If tests fail, read the error messages, understand the issue, and apply targeted patches. Repeat until tests pass.
11. AVOID UNRELATED MODIFICATIONS - Do not change code that is not related to the task at hand. Unrelated modifications introduce risk and complexity.
12. NEVER CLAIM SUCCESS WITHOUT VERIFICATION - Only finish when you have verified the result. Run tests, check the output, and confirm everything works.
13. EXPLICITLY FINISH WHEN THE TASK IS COMPLETE - Use the finish tool with a summary when the task is done. Do not leave the agent running unnecessarily.

YOU ARE OPERATING ON A REAL FILESYSTEM:
- All file operations remain within the designated workspace
- Path traversal (../) is prevented
- You cannot access files outside the workspace
- Treat the filesystem as real - never fabricate tool results"""