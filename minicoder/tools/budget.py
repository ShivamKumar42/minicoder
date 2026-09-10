"""Central context-budget limits for tool outputs.

Every tool that returns text to the LLM must cap its output so a single
call cannot flood the context window. Limits are deterministic head
truncation: the first N characters are kept and a note describing what
was omitted is appended, so the agent still knows more data exists and
how to narrow its next call.

Approximate sizing: ~4 characters per token, so 8000 chars ~= 2000
tokens per tool result, leaving room for multi-iteration history in
8k-128k context windows. File reads get a larger allowance (12000)
because source context is the agent's primary working material and
read_file already supports paging via start_line/end_line.
"""

from __future__ import annotations
from typing import Optional


# General tool-output cap (shell streams, search results, file listings,
# git output). ~= 2000 tokens per result.
DEFAULT_MAX_OUTPUT_CHARS = 8000

# File-content cap for read_file. ~= 3000 tokens; page with line ranges.
DEFAULT_MAX_FILE_CHARS = 12000

# File-listing caps: at most this many entries and this many characters.
DEFAULT_MAX_LIST_FILES = 5000
DEFAULT_MAX_LIST_CHARS = 20000

# Search caps (match the tool's historical behavior, now explicit).
DEFAULT_MAX_SEARCH_FILES = 50
DEFAULT_MAX_MATCHES_PER_FILE = 5
DEFAULT_MAX_MATCH_LINE_CHARS = 100

# Files larger than this are skipped by search (binary/minified blobs
# would otherwise be read fully into memory for zero benefit).
DEFAULT_MAX_SEARCH_FILE_BYTES = 1_000_000

# Total content budget for one search call across all scanned files.
# Bounds transient memory on huge repos; scanning stops deterministically
# (sorted file order) with an omission note when exhausted.
DEFAULT_MAX_SEARCH_TOTAL_BYTES = 20_000_000

# Read cap on line count: read_file pages through files instead of
# materializing them, and never returns more than this many lines.
DEFAULT_MAX_READ_LINES = 5000


def truncate_text(
    text: Optional[str],
    limit: int,
    hint: str = "",
) -> tuple[str, bool]:
    """Cap `text` at `limit` chars of original content (deterministic).

    Returns (possibly truncated text, was_truncated). When truncated,
    a note ``\\n... [truncated N chars]`` plus the optional hint is
    appended so the agent knows data was omitted and what to do next.
    At or under the limit the text is returned unchanged.
    """
    text = text or ""
    if limit < 0:
        limit = 0
    if len(text) <= limit:
        return text, False
    note = f"\n... [truncated {len(text) - limit} chars]"
    if hint:
        note += f" {hint}"
    return text[:limit] + note, True
