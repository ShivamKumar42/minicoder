from __future__ import annotations
import sys
import argparse
import signal
from typing import Dict, Any, Optional, List
import readline

from .config import get_settings, Settings
from .agent import Agent
from .llm.base import LLMClient
from .llm.openai import OpenAIClient
from .llm.anthropic import AnthropicClient
from .llm.openai_compatible import OpenAICompatibleClient
from .tools.registry import ToolRegistry
from .tools.filesystem import ListFilesTool
from .tools.read_file import ReadFileTool
from .tools.write_file import WriteFileTool
from .tools.apply_patch import ApplyPatchTool
from .tools.shell import RunCommandTool
from .tools.search import SearchFilesTool
from .tools.git import GitDiffTool, GitStatusTool
from .tools.finish import FinishTool
from .safety import SafetyTool
from .tracing import Tracer
from .messages import Message


class CLI:
    """Polished terminal CLI for MiniCoder."""
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self.agent: Optional[Agent] = None
        self.tracer: Optional[Tracer] = None
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog="minicoder",
            description="Autonomous coding agent built from first principles",
            formatter_class=argparse.RichHelpFormatter,
        )
        
        parser.add_argument(
            "task",
            nargs="?",
            type=str,
            default="",
            help="The coding task to perform",
        )
        
        parser.add_argument(
            "--workspace",
            type=str,
            default=None,
            help="Workspace directory (default: . or MINICODER_WORKSPACE)",
        )
        
        parser.add_argument(
            "--provider",
            type=str,
            default=None,
            help="LLM provider (openai, anthropic, openai-compatible)",
        )
        
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Model name",
        )
        
        parser.add_argument(
            "--base-url",
            type=str,
            default=None,
            help="Base URL for OpenAI-compatible APIs",
        )
        
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=None,
            help="Maximum iterations (default: 100 or MINICODER_MAX_ITERATIONS)",
        )
        
        parser.add_argument(
            "--max-tool-calls",
            type=int,
            default=None,
            help="Maximum tool calls (default: 200 or MINICODER_MAX_TOOL_CALLS)",
        )
        
        parser.add_argument(
            "--approval-mode",
            type=str,
            choices=["auto", "ask", "deny-dangerous"],
            default=None,
            help="Approval mode (default: auto or MINICODER_APPROVAL_MODE)",
        )
        
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=None,
            help="Dry-run mode: preview modifications without executing",
        )
        
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Enable verbose tracing",
        )
        
        parser.add_argument(
            "--env-file",
            type=str,
            default=".env",
            help="Environment file path",
        )
        
        return parser.parse_args(args)
    
    def create_provider(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
    ) -> LLMClient:
        """Create an LLM client based on provider."""
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAIClient(
                model=model,
                api_key=api_key,
                base_url=base_url if base_url else None,
            )
        elif provider == "anthropic":
            return AnthropicClient(
                model=model,
                api_key=api_key,
            )
        elif provider == "openai-compatible":
            return OpenAICompatibleClient(
                model=model,
                api_key=api_key,
                base_url=base_url or "http://localhost:11434/v1",
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with the given arguments."""
        parsed = self.parse_args(args)
        
        # Update settings from CLI args
        if parsed.workspace:
            self.settings.workspace = parsed.workspace
        if parsed.provider:
            self.settings.provider = parsed.provider
        if parsed.model:
            self.settings.model = parsed.model
        if parsed.base_url:
            self.settings.base_url = parsed.base_url
        if parsed.max_iterations is not None:
            self.settings.max_iterations = parsed.max_iterations
        if parsed.max_tool_calls is not None:
            self.settings.max_tool_calls = parsed.max_tool_calls
        if parsed.approval_mode:
            self.settings.approval_mode = parsed.approval_mode
        if parsed.dry_run or parsed.verbose is not None:
            self.settings.dry_run = parsed.dry_run
            self.settings.verbose = parsed.verbose
        
        # Set verbose from arg if provided
        if parsed.verbose is not None:
            self.settings.verbose = parsed.verbose
        
        # Initialize tracer
        self.tracer = Tracer(verbose=self.settings.verbose)
        
        # Get task
        task = parsed.task
        if not task:
            print("Error: No task provided. Use: minicoder 'your task here'")
            return 1
        
        # Print header
        self._print_header()
        
        try:
            # Run the agent
            result = self._run_agent(task)
            
            # Print result
            self._print_result(result)
            
            # Save traces if verbose
            if self.settings.verbose and self.tracer:
                trace_path = os.path.join(
                    self.settings.workspace,
                    ".minicoder_trace.json"
                )
                self.tracer.save_to_json(trace_path)
                print(f"\n[TRACE] Trace saved to: {trace_path}")
            
            return 0 if result.get("finished") else 1
            
        except KeyboardInterrupt:
            print("\n[AGENT] Interrupted by user")
            return 1
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {str(e)}")
            if self.settings.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _print_header(self) -> None:
        """Print the CLI header."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        
        console = Console()
        
        header_text = Text("MiniCoder", style="bold blue")
        header_text.append("\nAutonomous Coding Agent", style="")
        
        panel = Panel(
            header_text,
            subtitle="Provider: {self.settings.provider}, Model: {self.settings.model}",
            border_style="blue",
        )
        
        console.print(panel)
        console.print()
    
    def _print_result(self, result: dict[str, Any]) -> None:
        """Print the final result."""
        from rich.console import Console
        
        console = Console()
        
        if result.get("finished"):
            console.print("[green]✓[/green] Task completed successfully", style="")
        else:
            console.print("[red]✗[/red] Task did not complete", style="")
        
        if result.get("final_response"):
            console.print(f"\n[Agent] {result['final_response']}")
        
        if result.get("modified_files"):
            console.print(f"\n[Modified files]: {len(result['modified_files'])} files")
        
        if result.get("errors"):
            console.print(f"\n[Errors]: {len(result['errors'])} errors encountered")
            for err in result["errors"][:5]:
                console.print(f"  - {err}")
    
    def _run_agent(self, task: str) -> dict[str, Any]:
        """Run the agent loop."""
        from .agent import Agent
        
        settings = self.settings
        
        # Create LLM client
        api_key = settings.api_key or os.environ.get(
            {"openai": "OPENAI_API_KEY",
             "anthropic": "ANTHROPIC_API_KEY",
             "openai-compatible": "OPENAI_COMPATIBLE_API_KEY"}.get(
                 settings.provider, "OPENAI_API_KEY")
        )
        
        llm_client = self.create_provider(
            provider=settings.provider,
            model=settings.model,
            base_url=settings.base_url,
            api_key=api_key,
        )
        
        # Create tool registry
        registry = ToolRegistry()
        
        # Register all tools
        registry.register(ListFilesTool(workspace=settings.workspace))
        registry.register(ReadFileTool(workspace=settings.workspace))
        registry.register(WriteFileTool(workspace=settings.workspace))
        registry.register(ApplyPatchTool(workspace=settings.workspace))
        registry.register(RunCommandTool(workspace=settings.workspace, timeout=settings.timeout))
        registry.register(SearchFilesTool(workspace=settings.workspace))
        registry.register(GitDiffTool(workspace=settings.workspace))
        registry.register(GitStatusTool(workspace=settings.workspace))
        registry.register(FinishTool())
        registry.register(SafetyTool(workspace=settings.workspace))
        
        # Create agent
        self.agent = Agent(
            llm_client=llm_client,
            registry=registry,
            task=task,
            workspace=settings.workspace,
            max_iterations=settings.max_iterations,
            max_tool_calls=settings.max_tool_calls,
            approval_mode=settings.approval_mode,
            dry_run=settings.dry_run,
            tracer=self.tracer,
            verbose=settings.verbose,
        )
        
        # Run the agent loop
        return self.agent.run()


def main() -> None:
    """Entry point for the minicoder CLI."""
    import os
    
    cli = CLI()
    
    # Check for task in args or positional
    args = sys.argv[1:]
    
    # If no task and no args, show help
    if not args:
        print("Usage: minicoder --provider openai --model gpt-4o-mini 'your task here'")
        print("       minicoder 'Fix the failing tests'")
        sys.exit(1)
    
    exit_code = cli.run(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()