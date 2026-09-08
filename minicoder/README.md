# MiniCoder

**Autonomous coding agent built from first principles**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[Tests: 19/19 passing](https://github.com/ShivamKumar42/minicoder)

## Project Overview

MiniCoder is a compact, provider-agnostic autonomous coding agent built from first principles to explore the architecture behind modern AI coding agents. Unlike chatbots, MiniCoder demonstrates how an LLM becomes an agent through **iterative tool use, state management, observations, execution, and verification**.

The core intellectual goal is the agent loop:

```
USER TASK
   ↓
LLM
   ↓
TOOL CALL
   ↓
TOOL EXECUTION
   ↓
OBSERVATION
   ↓
LLM
   ↓
repeat
```

## Why Coding Agents Differ from Chatbots

A chatbot responds to a single prompt with a static answer. A coding agent:

1. **Iterates** - Makes multiple tool calls to gather information and verify results
2. **Maintains state** - Keeps track of task progress, modified files, and errors
3. **Uses tools deliberately** - Inspects, searches, reads, modifies, and runs commands
4. **Verifies outcomes** - Runs tests and treats command output as ground truth
5. **Knows when to stop** - Explicitly finishes when the task is complete with verification

This architecture demonstrates the fundamental mechanics of how modern AI coding agents (like Claude Code, OpenHands, etc.) work under the hood, without relying on framework magic.

## Architecture

```text
                    ┌──────────────┐
                    │    User      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Agent Loop  │
                    └──────┬───────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    LLM Provider   │
                 │ OpenAI / Claude / │
                 │ Local / Compatible│
                 └─────────┬─────────┘
                           │
                      tool calls
                           │
                           ▼
                    ┌──────────────┐
                    │ Tool Registry│
                    └──────┬───────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        filesystem       shell          git
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                       observation
                           │
                           └──────► LLM
```

### Core Components

| Component | Location | Description |
|-----------|----------|-------------|
| **Message Format** | `minicoder/messages.py` | Provider-neutral `Message`, `ToolCall`, `ToolResult` dataclasses |
| **LLM Client** | `minicoder/llm/` | Abstract base + OpenAI, Anthropic, OpenAI-compatible adapters |
| **Tool System** | `minicoder/tools/` | 9 tools with registry, schemas, and dispatch |
| **Agent State** | `minicoder/state.py` | Inspectable `AgentState` dataclass |
| **Agent Loop** | `minicoder/agent.py` | Explicit state machine with iteration/tool-call limits |
| **Safety** | `minicoder/safety.py` | Path traversal prevention, command blocking, dry-run |
| **Configuration** | `minicoder/config.py` | Environment variables, `.env` file, provider-specific vars |
| **CLI** | `minicoder/cli.py` | Rich-formatted terminal with all flags |
| **Prompts** | `minicoder/prompts.py` | Dedicated coding-agent system prompt |
| **Tracing** | `minicoder/tracing.py` | Structured tool/agent trace, JSON export |

## Architecture Diagram

```text
                    ┌──────────────┐
                    │    User      │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Agent Loop  │
                    └──────┬───────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    LLM Provider   │
                 │ OpenAI / Claude / │
                 │ Local / Compatible│
                 └─────────┬─────────┘
                           │
                      tool calls
                           │
                           ▼
                    ┌──────────────┐
                    │ Tool Registry│
                    └──────┬───────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        filesystem       shell          git
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                       observation
                           │
                           └──────► LLM
```

## Key Design Principles

- **Python 3.11+** with type hints
- **Clear interfaces** between components (agent, providers, tools, state)
- **Dependency injection** for LLM clients and tool registries
- **Explicit errors** rather than hidden framework exceptions
- **Minimal but meaningful abstractions** - no giant files, no giant functions
- **No hardcoded credentials** - all keys via environment variables
- **Provider-agnostic** core - agent code doesn't know or care about the underlying model
- **Testable** - 19 tests with `FakeLLMClient`, no API keys required

## Installation

```bash
# Clone the repository
git clone https://github.com/ShivamKumar42/minicoder.git
cd minicoder

# Install dependencies
pip install -e .

# Or install requirements
pip install pydantic rich pytest python-dotenv
```

## Configuration

Create a `.env` file in the project root:

```env
# Core provider settings
MINICODER_PROVIDER=openai
MINICODER_MODEL=gpt-4o-mini
MINICODER_BASE_URL=
MINICODER_API_KEY=

# Runtime settings
MINICODER_MAX_ITERATIONS=100
MINICODER_MAX_TOOL_CALLS=200
MINICODER_TIMEOUT=60
MINICODER_DRY_RUN=false
MINICODER_APPROVAL_MODE=auto
MINICODER_VERBOSE=false

# Workspace
MINICODER_WORKSPACE=.

# Provider-specific environment variable mappings:
# OpenAI:    MINICODER_OPENAI_API_KEY, MINICODER_OPENAI_MODEL, MINICODER_OPENAI_BASE_URL
# Anthropic: MINICODER_ANTHROPIC_API_KEY, MINICODER_ANTHROPIC_MODEL
# Compatible: MINICODER_OPENAI_COMPATIBLE_API_KEY, MINICODER_OPENAI_COMPATIBLE_MODEL, MINICODER_OPENAI_COMPATIBLE_BASE_URL
```

### Provider Configuration

| Provider | Environment Variables |
|----------|----------------------|
| **OpenAI** | `MINICODER_PROVIDER=openai`, `MINICODER_MODEL=...`, `MINICODER_API_KEY=...` |
| **Anthropic** | `MINICODER_PROVIDER=anthropic`, `MINICODER_MODEL=...`, `MINICODER_API_KEY=...` |
| **OpenAI-compatible** | `MINICODER_PROVIDER=openai-compatible`, `MINICODER_MODEL=...`, `MINICODER_API_KEY=...`, `MINICODER_BASE_URL=http://localhost:11434/v1` |

## Supported Providers

### OpenAI

```bash
MINICODER_PROVIDER=openai
MINICODER_MODEL=gpt-4o-mini
MINICODER_API_KEY=sk-...
```

### Anthropic / Claude

```bash
MINICODER_PROVIDER=anthropic
MINICODER_MODEL=claude-3-haiku-20240307
MINICODER_API_KEY=...
```

### OpenAI-compatible (Local/Cloud)

```bash
MINICODER_PROVIDER=openai-compatible
MINICODER_MODEL=llama2
MINICODER_API_KEY=ollama  # Often not required for local
MINICODER_BASE_URL=http://localhost:11434/v1  # Ollama, LM Studio, vLLM, etc.
```

### OpenRouter, Ollama, LM Studio, Together AI, Groq, vLLM

These all work through the `openai-compatible` provider setting with the appropriate `BASE_URL`.

## CLI Usage

```bash
# Basic usage with task
minicoder "Fix the failing tests"

# With workspace
minicoder "Fix the authentication bug" --workspace ./my-project

# With provider and model
minicoder --provider openai --model gpt-4o-mini "Fix the failing tests"

minicoder --provider anthropic --model claude-3-haiku-20240307 "Fix the failing tests"

minicoder --provider openai-compatible \
    --base-url http://localhost:11434/v1 \
    --model llama2 "Fix the failing tests"

# Control flow
minicoder --max-iterations 50 "Fix the bug"
minicoder --max-tool-calls 100 "Fix the bug"
minicoder --approval-mode ask "Fix the bug"       # Ask before dangerous ops
minicoder --approval-mode auto "Fix the bug"      # Auto-approve
minicoder --approval-mode deny-dangerous "Fix the bug"  # Block dangerous only
minicoder --dry-run "Inspect the code"          # Preview only, no modifications
minicoder --verbose "Fix the bug"               # Detailed tracing
minicoder --env-file .env "Fix the bug"         # Custom env file
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `task` | - | The coding task to perform (positional arg) |
| `--workspace` | `.` | Workspace directory |
| `--provider` | `openai` | LLM provider (openai, anthropic, openai-compatible) |
| `--model` | `gpt-4o-mini` | Model name |
| `--base-url` | `""` | Base URL for OpenAI-compatible APIs |
| `--max-iterations` | `100` | Maximum agent iterations |
| `--max-tool-calls` | `200` | Maximum tool calls per run |
| `--approval-mode` | `auto` | Approval mode (auto, ask, deny-dangerous) |
| `--dry-run` | `false` | Preview mode: no file modifications or side effects |
| `--verbose` | `false` | Enable detailed tracing output |
| `--env-file` | `.env` | Environment file path |

### Output Formatting

The CLI uses **rich** for terminal formatting with these visual elements:

- `[Agent]` - Agent iteration display
- `[Tool]` - Tool execution display
- `[Result]` - Tool result display
- `[Test]` - Test result display
- `[Error]` - Error display
- `[Finished]` - Task completion display

## Testing

All tests use a `FakeLLMClient` that simulates a genuine coding-agent trajectory without requiring API keys:

```bash
# Run the full test suite
python -m pytest minicoder/tests/ -v

# Output: 19 passed in ~10s
```

The test coverage includes:

- **Provider abstraction** - Test the LLM client base class and adapters
- **Tool registry** - Register/lookup/list tools and schemas
- **Tool schemas** - Validate parameter schemas for each tool
- **Tool dispatch** - Test tool execution through the registry
- **Path traversal protection** - Block `../` and absolute paths
- **File reading** - Read files with line ranges, error handling
- **File writing** - Create/write files within workspace boundaries
- **Patch application** - Apply unified diff hunk patches
- **Search** - Text/regex search across the workspace
- **Command timeout** - Test command execution with timeouts
- **Dangerous command blocking** - Block sudo, shutdown, rm -rf, etc.
- **Approval mode** - Test auto, ask, and deny-dangerous modes
- **Dry-run** - Preview mode verification
- **Malformed tool calls** - Handle unknown tools and invalid arguments
- **Unknown tools** - Graceful handling of unrecognized tool names
- **Provider errors** - Test configuration and API key handling
- **Iteration limits** - Enforce max_iterations boundary
- **Tool-call limits** - Enforce max_tool_calls boundary
- **Finish behavior** - Verify finish tool execution

Run specific test categories:

```bash
# Messages tests
python -m pytest minicoder/tests/test_agent.py::TestMessages -v

# Agent loop tests
python -m pytest minicoder/tests/test_agent.py::test_agent_loop_with_fake_llm -v

# Fake LLM trajectory
python -m pytest minicoder/tests/test_agent.py::TestFakeLLM -v
```

## Example Execution Trace

```text
[Agent] Iteration 1
[TOOL] list_files: OK - Listing workspace files
[Result] 12 files found in workspace
[TOOL] read_file: OK - Reading calculator.py
[Result] File contents displayed
[TOOL] search_files: OK - Searching for "add" pattern
[Result] Found 3 matches in calculator.py
[TOOL] apply_patch: OK - Applied patch to calculator.py
[Result] Patch applied successfully
[TOOL] run_command: OK - Running pytest
[Test] 3 tests passed
[AGENT] Iteration 2
[TOOL] finish: OK - Task complete
[Finished] Task completed successfully in 2 iterations with 5 tool calls
```

## Demo Project

MiniCoder comes with a realistic demo project that introduces intentional defects, forcing the agent to:

1. **Inspect files** - Read calculator.py and validators.py
2. **Search the codebase** - Find the `add` and `validate` patterns
3. **Understand tests** - Read test_Calculator.py and test_validators.py to understand expected behavior
4. **Patch code** - Apply targeted fixes using `apply_patch`
5. **Run tests** - Verify fixes with `run_command`
6. **Analyze failures** - When tests fail, investigate and iterate
7. **Patch again** - Apply second fix for remaining defects
8. **Rerun tests** - Confirm all tests pass
9. **Verify all tests pass** - Final verification before finishing
10. **Finish** - Provide summary with tests_run and remaining_issues=0

### Demo Project Structure

```
demo_project/
    app/
        calculator.py    # Buggy: add() subtracts, subtract() adds
        validators.py    # Buggy: wrong email/password/username validation
    tests/
        test_calculator.py    # Tests exposing calculator bugs
        test_validators.py    # Tests exposing validator bugs
    README.md           # Demo project summary
```

The agent successfully fixed both defects through the iterative loop, demonstrating the complete agent trajectory.

## Limitations

- **No persistent memory** - Each run starts fresh (state is not saved between runs)
- **Simple patch application** - The `apply_patch` tool uses a simplified unified diff parser; complex patches may need manual adjustment
- **Command safety** - Blocking is based on keyword detection, not full semantic analysis
- **Provider capabilities** - Not all models support native tool calling; the architecture normalizes this but some capabilities are lost
- **Workspace scope** - All operations are confined to the specified workspace directory
- **Single-task focus** - The agent handles one task at a time; multi-tasking is not supported

## Future Improvements

Optional advanced features that can be added cleanly:

- **Streaming model responses** - Real-time token-by-token output
- **Run history** - Persist agent runs for later inspection
- **Token/context statistics** - Track token usage and context window consumption
- **Automatic git diff summary** - Summarize changes with git diff
- **Repository map/tree summary** - Generate overview of project structure
- **Context compaction** - Summarize long conversations to fit context windows
- **Configurable system prompts** - Allow users to customize the agent's instructions
- **Session persistence** - Save and restore agent state between runs
- **Patch rollback** - Revert unsuccessful patches
- **Task checkpoints** - Save progress at intermediate points

## License

MIT - See the LICENSE file for details.

---

*"The important implementation is not the model itself. The project demonstrates how an LLM becomes an agent through iterative tool use, state, observations, execution, and verification."*