# MiniCoder

**An autonomous coding agent built from first principles — no framework magic.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 145 passing](https://img.shields.io/badge/tests-145%20passing-brightgreen.svg)](#testing)

MiniCoder shows how an LLM becomes a coding agent: an explicit loop of **tool calls, execution, observation, and verification**, with provider-agnostic messaging, persistent sessions, safety guardrails, and 145 hermetic tests that run without API keys.

```
USER TASK ─▶ LLM ─▶ TOOL CALL ─▶ EXECUTION ─▶ OBSERVATION ─▶ LLM ─▶ … ─▶ VERIFIED DONE
```

## Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Tools](#tools)
- [Safety and security](#safety-and-security)
- [Sessions](#sessions)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Demo project](#demo-project)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Provider-agnostic core** — OpenAI, Anthropic, and any OpenAI-compatible API (Ollama, LM Studio, vLLM, Together, Groq, OpenRouter) behind one `Message` / `ToolCall` interface with strict JSON argument parsing.
- **Real agent loop** — bounded iterations and tool calls, malformed-response handling, approval modes, dry-run previews, and structured tracing.
- **Verifiable behavior** — the agent must run tests and call `finish`; tool failures are surfaced, never silently swallowed.
- **Persistent sessions** — save, resume, and continue interrupted runs from a plain-JSON session file (no database, no secrets stored).
- **Context budgets** — deterministic truncation caps on every tool output, paged file reads, and bounded history.
- **Confinement by default** — realpath workspace guard (symlink-aware), shell cwd confinement, command blocklists with shell-quoting normalization.
- **145 tests, no API keys** — scripted-LLM trajectories plus focused regression suites; the full suite runs in about a second.

## Quickstart

```bash
git clone https://github.com/ShivamKumar42/minicoder.git
cd minicoder

pip install -e .
# ...or install the runtime deps directly:
pip install pydantic rich pytest python-dotenv
```

Configure credentials (never commit this file):

```bash
cp .env.example .env   # then fill in your key
```

```env
MINICODER_PROVIDER=openai
MINICODER_MODEL=gpt-4o-mini
MINICODER_API_KEY=sk-...
```

Run your first task:

```bash
minicoder --workspace ./my-project "Fix the failing tests"
```

## Usage

```bash
# Basic task
minicoder "Fix the failing tests"

# Scoped to a project, specific provider and model
minicoder --workspace ./my-project --provider anthropic \
    --model claude-3-haiku-20240307 "Fix the authentication bug"

# Local models (Ollama, LM Studio, vLLM, …)
minicoder --provider openai-compatible \
    --base-url http://localhost:11434/v1 \
    --model llama2 "Fix the failing tests"

# Safety and control
minicoder --approval-mode ask "Refactor the login flow"
minicoder --approval-mode deny-dangerous "Update dependencies"
minicoder --dry-run "Inspect the codebase"      # preview only, no writes/commands
minicoder --verbose "Fix the bug"               # tracing + trace JSON export

# Long-running work you can interrupt and resume
minicoder --session run1.json "Migrate the test suite"
minicoder --session run1.json --resume           # continue where it stopped

# Budgets and custom env files
minicoder --max-iterations 50 --max-tool-calls 100 "Fix the bug"
minicoder --max-tool-output-chars 4000 --max-history 60 "Fix the bug"
minicoder --env-file .env.prod "Fix the bug"
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `task` | — | The coding task (positional; optional when resuming) |
| `--workspace` | `.` | Workspace directory (`MINICODER_WORKSPACE`) |
| `--provider` | `openai` | `openai`, `anthropic`, or `openai-compatible` |
| `--model` | `gpt-4o-mini` | Model name |
| `--base-url` | `""` | Base URL for OpenAI-compatible APIs |
| `--max-iterations` | `100` | Maximum agent iterations |
| `--max-tool-calls` | `200` | Maximum tool calls per run |
| `--max-tool-output-chars` | `8000` | Chars kept per tool result |
| `--max-history` | `100` | Messages kept in agent history |
| `--approval-mode` | `auto` | `auto`, `ask`, or `deny-dangerous` |
| `--dry-run` | off | Skip `write_file`, `apply_patch`, `run_command` |
| `--verbose` | off | Detailed tracing + `.minicoder_trace.json` export |
| `--session` | — | Session file to save to (or load with `--resume`) |
| `--resume` | off | Resume the `--session` file (requires `--session`) |
| `--env-file` | `.env` | Custom environment file path |

## Configuration

All settings come from the environment (optionally via `.env` — see [`.env.example`](.env.example)); CLI flags override them.

| Variable | Default | Description |
|----------|---------|-------------|
| `MINICODER_PROVIDER` | `openai` | LLM provider |
| `MINICODER_MODEL` | `gpt-4o-mini` | Model name |
| `MINICODER_BASE_URL` | `""` | Base URL for compatible APIs |
| `MINICODER_API_KEY` | `""` | API key (preferred) |
| `MINICODER_{PROVIDER}_API_KEY` | `""` | Provider-specific key, e.g. `MINICODER_OPENAI_API_KEY` (fallback) |
| `MINICODER_MAX_ITERATIONS` | `100` | Iteration budget |
| `MINICODER_MAX_TOOL_CALLS` | `200` | Tool-call budget |
| `MINICODER_TIMEOUT` | `60` | Shell-command timeout (seconds, positive) |
| `MINICODER_MAX_TOOL_OUTPUT_CHARS` | `8000` | Output cap per tool result (positive) |
| `MINICODER_MAX_HISTORY` | `100` | History bound (positive) |
| `MINICODER_APPROVAL_MODE` | `auto` | `auto`, `ask`, `deny-dangerous` |
| `MINICODER_DRY_RUN` | `false` | Preview mode |
| `MINICODER_VERBOSE` | `false` | Verbose tracing |
| `MINICODER_WORKSPACE` | `.` | Workspace directory |

Legacy `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_COMPATIBLE_API_KEY` variables are still honored as a final fallback.

## Architecture

```text
                ┌──────────────┐
                │     User     │
                └──────┬───────┘
                       ▼
                ┌──────────────┐      ┌───────────────┐
                │  Agent Loop  │─────▶│ Session file  │
                │  (bounded,   │      │ (save/resume) │
                │   traced)    │      └───────────────┘
                └──────┬───────┘
                       ▼ tools (internal schema)
          ┌─────────────────────────┐
          │  Provider converters    │
          │  OpenAI / Anthropic     │
          └────────────┬────────────┘
                       ▼
                ┌──────────────┐
                │ Tool Registry│
                └──────┬───────┘
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
    filesystem        shell     search/git/patch
         │             │              │
         └─────────────┴──────────────┘
                       ▼ observation (budget-capped)
                       └──────▶ LLM …
```

| Component | Location | Description |
|-----------|----------|-------------|
| Agent loop | `minicoder/agent.py` | Bounded iterations, response handling, approvals, dry-run, history cap, tracer feed |
| Agent state | `minicoder/state.py` | Serializable `AgentState` + JSON session envelopes |
| Messages | `minicoder/messages.py` | Provider-neutral `Message` / `ToolCall` / `ToolResult` |
| LLM clients | `minicoder/llm/` | Abstract base, shared OpenAI-compatible core, Anthropic adapter, schema converters |
| Tools | `minicoder/tools/` | Registry, path guard, context budgets, 10 tools |
| Safety | `minicoder/safety.py` | Command blocklists, quoting normalization, workspace checks |
| Config | `minicoder/config.py` | Table-driven env parsing, validation |
| CLI | `minicoder/cli.py` | Rich terminal, sessions, budgets, tracing export |
| Prompts | `minicoder/prompts.py` | Single system-prompt source |
| Tracing | `minicoder/tracing.py` | Per-iteration tool traces, JSON export |

## Tools

| Tool | Description |
|------|-------------|
| `list_files` | Workspace listing (recursive optional, budget-capped) |
| `read_file` | Paged reads with line ranges (never materializes huge files) |
| `write_file` | Create/overwrite files inside the workspace |
| `apply_patch` | Exact-snippet replacement (must match exactly once) |
| `run_command` | Shell with timeout, truncation, and safety policy |
| `search_files` | Regex search with per-file/total byte budgets, deterministic order |
| `git_diff` / `git_status` | Repository inspection (path-scoped, truncated) |
| `finish` | Explicit completion with summary and test counts |
| `safety_check` | Path/command safety queries |

## Safety and security

- **Workspace confinement** — all file tools resolve symlinks (`realpath`) and reject escapes; shell working directories are confined the same way.
- **Command policy** — blocklists with whole-word matching and shell-quoting normalization (`su\do`, `s""udo` are caught; `address` is not falsely blocked).
- **Human control** — `ask` approval (EOF-safe), `deny-dangerous` mode, and `dry-run` previews that skip writes and commands.
- **Honest limits of the model** — `run_command` is inherently fully privileged (e.g. `python3 -c …` needs no blocked token), so the blocklist is best-effort defense in depth, not a sandbox. Review generated commands before running untrusted tasks.

## Sessions

```bash
minicoder --session run1.json "Migrate the test suite"
# ^C to interrupt — progress is saved, then:
minicoder --session run1.json --resume
```

The session file is plain JSON (`version` + `state` + runtime counters): message history with tool-call correlation, iterations, executed tools, modified files, test results, errors, and completion state. It never contains API keys; missing/corrupt/incompatible files fail with a clear error instead of crashing.

## Testing

No API keys needed — scripted LLMs drive the agent through realistic trajectories, and the integration test runs hermetically in a temp workspace (no subprocesses, no hardcoded paths).

```bash
# From the repo root
python -m pytest minicoder/tests/ -v
# 145 passed in ~1s
```

| Suite | Tests | What it covers |
|-------|-------|----------------|
| `test_agent.py` | 19 | Messages, state, registry, config, safety, prompts, hermetic agent loop |
| `test_hardening.py` | 29 | Loop bounds, history validity, dry-run, approvals, sessions, CLI flags, tracer |
| `test_budget.py` | 24 | Truncation boundaries, determinism, per-tool caps, env config |
| `test_confinement.py` | 15 | Symlink escape, quoting bypasses, cwd confinement, budgets |
| `test_llm_parsing.py` | 12 | Strict JSON parsing, malformed input, no-code-execution proof |
| `test_agent_correctness.py` | 10 | Task delivery, dict/object responses, completion semantics |
| `test_tool_schemas.py` | 10 | OpenAI/Anthropic conversion, round trips |
| `test_apply_patch.py` | 9 | Replacement semantics and every failure mode |
| `test_session.py` | 9 | Save → load → continue, corruption handling |
| `test_shell.py` | 8 | Pipes, chaining, truncation, blocking |

## Project structure

```
.
├── README.md                  # this file
├── LICENSE                    # MIT
├── pyproject.toml             # packaging, entry point, pytest config
├── .env.example               # documented configuration template
└── minicoder/
    ├── agent.py               # agent loop
    ├── state.py               # serializable state + sessions
    ├── messages.py            # Message / ToolCall / ToolResult
    ├── prompts.py             # system prompt (single source)
    ├── safety.py              # blocklists + guards
    ├── config.py              # settings
    ├── cli.py                 # terminal interface
    ├── tracing.py             # run traces
    ├── llm/                   # base, openai_base, openai,
    │                          # openai_compatible, anthropic, tool_schemas
    ├── tools/                 # registry, base, paths, budget,
    │                          # filesystem, read_file, write_file,
    │                          # apply_patch, shell, search, git, finish
    ├── tests/                 # 145 tests + scripted FakeLLMClient
    └── demo_project/          # buggy sample app + tests for manual runs
```

## Demo project

`minicoder/demo_project/` is a small app with intentional defects (wrong arithmetic, lax validators) plus failing tests — a realistic manual target for the agent:

```bash
minicoder --workspace minicoder/demo_project "Fix the failing tests"
```

## Limitations

- `run_command` is powerful by design; treat generated commands as privileged (see above).
- Search skips files over 1 MB and caps total scanning (configurable); history keeps the latest 100 messages by default.
- `ask` approval needs an interactive terminal (non-interactive stdin safely denies).
- One task per run; no streaming responses or token-usage accounting yet.
- Not all local models support native tool calling; capability varies by provider.

## Roadmap

- [ ] Token/context usage statistics
- [ ] Patch rollback and task checkpoints
- [ ] Repository map for large codebases
- [ ] Custom system prompts via config
- [ ] Richer session inspection commands

## Contributing

Issues and pull requests are welcome at [ShivamKumar42/minicoder](https://github.com/ShivamKumar42/minicoder). Please keep changes small and scoped, add regression tests for fixes, and run the suite (`python -m pytest minicoder/tests/ -q`) before pushing. Report security concerns privately rather than in public issues.

## License

MIT — see [LICENSE](LICENSE) for details.

---

*MiniCoder demonstrates the core idea behind modern coding agents: the model matters less than the loop — iterate with tools, observe ground truth, verify before finishing.*
