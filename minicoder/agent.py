from __future__ import annotations
from typing import Any, List, Optional

from .messages import Message, ToolCall, ToolResult
from .prompts import SYSTEM_PROMPT
from .safety import APPROVAL_DANGEROUS_KEYWORDS, contains_dangerous_keyword
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
        state: Optional[AgentState] = None,
        total_tool_calls: int = 0,
        max_history: int = 100,
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
        self.max_history = max_history

        # Resume a saved run when a state is provided; otherwise start fresh.
        if state is not None:
            self.state = state
            self.task = state.task
            self.workspace = state.workspace
        else:
            # Initialize state
            self.state = AgentState(
                task=task,
                workspace=workspace,
                max_iterations=max_iterations,
            )

            # Seed the conversation with the user's task so the LLM
            # always receives the actual task/workspace in generate().
            self.state.add_message(
                Message(
                    role="user",
                    content=f"Task: {task}\nWorkspace: {workspace}",
                )
            )

        # Tool call tracking
        self.total_tool_calls = total_tool_calls
        self.tool_call_history: List[dict[str, Any]] = []

    @classmethod
    def from_session(
        cls,
        llm_client: Any,
        registry: ToolRegistry,
        path: str,
        approval_mode: str = "auto",
        dry_run: bool = False,
        tracer: Optional[Any] = None,
        verbose: bool = False,
        max_tool_calls: int = 200,
        max_history: int = 100,
    ) -> Agent:
        """Restore an agent from a session file written by `save_session`."""
        state, total_tool_calls = AgentState.load(path)
        return cls(
            llm_client=llm_client,
            registry=registry,
            task=state.task,
            workspace=state.workspace,
            max_iterations=state.max_iterations,
            max_tool_calls=max_tool_calls,
            approval_mode=approval_mode,
            dry_run=dry_run,
            tracer=tracer,
            verbose=verbose,
            state=state,
            total_tool_calls=total_tool_calls,
            max_history=max_history,
        )

    # Tools that can modify state; skipped (not executed) in dry-run mode.
    DRY_RUN_SKIPPED_TOOLS = frozenset({"write_file", "apply_patch", "run_command"})

    # Error prefix convention used by mutating/status tools. Tool outputs
    # with this prefix are recorded in state.errors so failures are
    # visible in results instead of silently passing as observations.
    # Data-returning tools (read/search/list/git) are exempt: their
    # payload is file content, which may legitimately start with it.
    TOOL_ERROR_PREFIX = "Error:"
    READ_ONLY_TOOLS = frozenset(
        {"read_file", "search_files", "list_files", "git_diff", "git_status"}
    )

    # Tools whose successful execution modifies files; their target paths
    # are recorded in state.modified_files. Maps tool name -> path argument.
    FILE_MODIFYING_TOOLS = {"write_file": "path", "apply_patch": "filepath"}

    def save_session(self, path: str) -> str:
        """Persist current state + tool-call count (no secrets)."""
        return self.state.save(path, total_tool_calls=self.total_tool_calls)
    
    def run(self) -> dict[str, Any]:
        """Run the agent loop until completion."""
        # Continue from the saved iteration when resuming a session.
        iteration = self.state.current_iteration
        consecutive_llm_errors = 0

        while self.state.can_proceed():
            iteration += 1
            self.state.current_iteration = iteration

            if self.verbose:
                print(f"[AGENT] Iteration {iteration}")

            self._trace_start_iteration(iteration)

            # Generate response from LLM
            try:
                response = self._generate_response()
            except Exception as e:
                self.state.add_error(f"LLM generation error: {str(e)}")
                if self.verbose:
                    print(f"[AGENT] LLM generation error: {e}")
                # A failed generation still consumes an iteration so a
                # persistently failing LLM cannot spin forever; the loop
                # remains bounded by max_iterations.
                consecutive_llm_errors += 1
                if consecutive_llm_errors >= 3:
                    self.state.add_error(
                        "Aborting after 3 consecutive LLM generation errors"
                    )
                    break
                continue
            consecutive_llm_errors = 0

            # Process response
            try:
                self._process_response(response)
            except Exception as e:
                self.state.add_error(f"Response handling error: {str(e)}")
                if self.verbose:
                    print(f"[AGENT] Response handling error: {e}")
                continue

            self._trace_record_iteration()

            # Check if we should continue
            if self.state.finished:
                break

            # Check tool call limits
            if self.total_tool_calls >= self.max_tool_calls:
                self.state.add_error(
                    f"Maximum tool calls ({self.max_tool_calls}) reached"
                )
                break

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
        tool_calls = self._extract_tool_calls(response)
        content = self._extract_content(response)

        if tool_calls:
            if not isinstance(tool_calls, (list, tuple)):
                self.state.add_error(
                    f"Malformed tool_calls in LLM response "
                    f"(expected list, got {type(tool_calls).__name__}); ignoring"
                )
                return
            # Normalize dict-style tool calls to ToolCall objects.
            normalized: List[ToolCall] = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    args = tc.get("arguments", {})
                    normalized.append(
                        ToolCall(
                            id=str(tc.get("id", "")),
                            name=str(tc.get("name", "")),
                            arguments=args if isinstance(args, dict) else {},
                        )
                    )
                elif isinstance(tc, ToolCall):
                    normalized.append(tc)
                else:
                    self.state.add_error(
                        f"Malformed tool call entry "
                        f"({type(tc).__name__}); skipping"
                    )
                    continue
            if not normalized:
                return
            # Record the assistant's tool-call request so provider
            # histories stay valid (assistant -> tool result order).
            self._append_message(
                Message(
                    role="assistant",
                    content=content,
                    tool_calls=list(normalized),
                )
            )
            # Execute tool calls and continue the loop (do not finish).
            for tool_call in normalized:
                errors_before = len(self.state.errors)
                self._trace_start_tool(tool_call)
                self._execute_tool_call(tool_call)
                self._trace_end_tool(
                    success=len(self.state.errors) == errors_before,
                    error=self.state.errors[-1]
                    if len(self.state.errors) > errors_before
                    else None,
                )

            # Append tool results to messages
            self._append_tool_results(normalized)

        elif content and content.strip():
            # Model produced a final response without tool calls
            self.state.final_response = content
            self.state.finished = True

            if self.verbose:
                print(f"[AGENT] Final response: {content[:100]}...")
        else:
            # Neither tool calls nor text: record so it is not
            # silently discarded (agent will continue/retry).
            self.state.add_error(
                "Empty response from LLM (no content or tool calls)"
            )

    @staticmethod
    def _extract_tool_calls(response: Any) -> List[Any]:
        """Support both dict-style and object-style responses."""
        if isinstance(response, dict):
            return response.get("tool_calls") or []
        return getattr(response, "tool_calls", None) or []

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Support both dict-style and object-style text content."""
        if isinstance(response, dict):
            content = response.get("content")
            if content is None:
                content = response.get("text")
            return content if isinstance(content, str) else (str(content) if content is not None else "")
        content = getattr(response, "content", None)
        if content is None:
            content = getattr(response, "text", None)
        return content if isinstance(content, str) else (str(content) if content is not None else "")
    
    def _execute_tool_call(self, tool_call: ToolCall) -> None:
        """Execute a single tool call."""
        tool_name = tool_call.name
        tool_args = (
            tool_call.arguments
            if isinstance(tool_call.arguments, dict)
            else {}
        )

        self.total_tool_calls += 1
        # Persist tool-execution history in state so resumed runs keep it.
        self.state.tool_calls_executed.append(tool_call)

        # Look up the tool
        tool = self.registry.lookup(tool_name)
        if tool is None:
            error_msg = f"Unknown tool: {tool_name}"
            self.state.add_error(error_msg)
            self._append_message(
                Message(
                    role="tool",
                    content=error_msg,
                    tool_call_id=tool_call.id,
                    tool_calls=[tool_call],
                )
            )
            return

        # Check approval mode
        if not self._check_approval(tool_name, tool_args):
            self.state.add_error(f"Tool call denied by approval mode: {tool_name}")
            self._append_message(
                Message(
                    role="tool",
                    content=f"Tool call denied by approval mode: {tool_name}",
                    tool_call_id=tool_call.id,
                )
            )
            return

        # Dry-run mode: preview only, no side effects.
        if self.dry_run and tool_name in self.DRY_RUN_SKIPPED_TOOLS:
            self._append_message(
                Message(
                    role="tool",
                    content=(
                        f"Dry run: skipped {tool_name} with args "
                        f"{str(tool_args)[:200]} (no changes made)"
                    ),
                    tool_call_id=tool_call.id,
                )
            )
            return

        # Execute the tool
        try:
            result = tool.execute(**tool_args)

            # Success path
            output = result if isinstance(result, str) else str(result)
            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                output=output,
                success=True,
            )

            self._append_message(
                Message(
                    role="tool",
                    content=tool_result.output,
                    tool_call_id=tool_call.id,
                )
            )

            # Tool outputs follow the "Error:" convention on failure;
            # surface those in state.errors instead of silently passing.
            if output.startswith(self.TOOL_ERROR_PREFIX) and (
                tool_name not in self.READ_ONLY_TOOLS
            ):
                self.state.add_error(f"Tool {tool_name} failed: {output[:200]}")
            elif tool_name in self.FILE_MODIFYING_TOOLS:
                target = tool_args.get(self.FILE_MODIFYING_TOOLS[tool_name], "")
                if target and target not in self.state.modified_files:
                    self.state.modified_files.append(str(target))

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
                    tool_call_id=tool_call.id,
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
                if contains_dangerous_keyword(cmd, APPROVAL_DANGEROUS_KEYWORDS):
                    print(
                        f"[APPROVAL] Denied dangerous command: {cmd[:80]}...")
                    return False
            return True
        elif self.approval_mode == "ask":
            print(f"[APPROVAL] Approve tool {tool_name} with args {str(tool_args)[:100]}?")
            try:
                response = input("[y/n/q] ").strip().lower()
            except EOFError:
                # Non-interactive stdin (piped/CI): deny safely instead of
                # crashing the loop with an uncaught exception.
                print("[APPROVAL] No input available; denying by default")
                return False
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
    
    def _trace_start_iteration(self, iteration: int) -> None:
        """Feed the tracer; tracing must never break the agent loop."""
        if self.tracer is None:
            return
        try:
            model = getattr(self.llm_client, "model", "") or ""
            provider = getattr(self.llm_client, "provider", "") or ""
            if not isinstance(model, str):
                model = str(model)
            if not isinstance(provider, str):
                provider = str(provider)
            self.tracer.start_iteration(iteration, model, provider)
        except Exception:
            pass

    def _trace_record_iteration(self) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.record_agent_iteration()
        except Exception:
            pass

    def _trace_start_tool(self, tool_call: ToolCall) -> None:
        if self.tracer is None:
            return
        try:
            args = (
                tool_call.arguments
                if isinstance(tool_call.arguments, dict)
                else {}
            )
            self.tracer.start_tool(tool_call.name, args)
        except Exception:
            pass

    def _trace_end_tool(
        self, success: bool, error: Optional[str] = None
    ) -> None:
        if self.tracer is None:
            return
        try:
            self.tracer.end_tool(success=success, error=error)
        except Exception:
            pass

    def _append_tool_results(self, tool_calls: List[ToolCall]) -> None:
        """Append tool results to messages (called after execution)."""
        # Tool results are appended in _execute_tool_call
        pass
    
    def _append_message(self, message: Message) -> None:
        """Append a message to the state (bounding history length)."""
        self.state.add_message(message)
        # Bound memory/context: keep the seed message plus the most recent
        # entries so long runs cannot grow history without limit.
        overflow = len(self.state.messages) - self.max_history
        if overflow > 0 and len(self.state.messages) > 1:
            del self.state.messages[1 : 1 + overflow]