"""Regression tests for LLM tool-call argument parsing security fix.

Verifies that llm/openai.py and llm/openai_compatible.py use strict
JSON parsing (json.loads) instead of eval():
- valid JSON parses correctly (including JSON literals eval() choked on)
- malformed JSON is rejected safely with a useful error
- arbitrary Python expressions are NEVER evaluated
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minicoder.llm.openai import OpenAIClient
from minicoder.llm.openai_compatible import OpenAICompatibleClient


def _fake_tc(name: str, arguments: object, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


CLIENTS = [
    ("openai", OpenAIClient(model="test-model", api_key="test-key")),
    (
        "openai-compatible",
        OpenAICompatibleClient(model="test-model", api_key="test-key"),
    ),
]


def _parse(client: object, tool_calls: object):  # type: ignore[no-untyped-def]
    if isinstance(client, OpenAIClient):
        return client._parse_openai_tool_calls(tool_calls)
    return client._parse_tool_calls(tool_calls)  # type: ignore[union-attr]


@pytest.mark.parametrize("provider,client", CLIENTS)
def test_valid_json_parses_correctly(provider: str, client: object) -> None:
    # JSON literals (true/null/numbers) that eval() failed on must work.
    raw = json.dumps({"path": ".", "recursive": True, "limit": 5, "opt": None})
    parsed = _parse(client, [_fake_tc("list_files", raw)])
    assert len(parsed) == 1
    assert parsed[0].id == "call_1"
    assert parsed[0].name == "list_files"
    assert parsed[0].arguments == {
        "path": ".",
        "recursive": True,
        "limit": 5,
        "opt": None,
    }


@pytest.mark.parametrize("provider,client", CLIENTS)
def test_malformed_json_rejected_safely(provider: str, client: object) -> None:
    with pytest.raises(ValueError, match="Invalid tool arguments JSON"):
        _parse(client, [_fake_tc("read_file", '{"path": }')])


@pytest.mark.parametrize("provider,client", CLIENTS)
def test_non_object_json_rejected(provider: str, client: object) -> None:
    with pytest.raises(ValueError, match="must decode to an object"):
        _parse(client, [_fake_tc("read_file", '[1, 2, 3]')])


@pytest.mark.parametrize("provider,client", CLIENTS)
def test_python_expressions_not_evaluated(provider: str, client: object) -> None:
    # Would return 3 under eval(); must be rejected as invalid JSON.
    with pytest.raises(ValueError, match="Invalid tool arguments JSON"):
        _parse(client, [_fake_tc("run_command", "1+1")])

    # Valid Python dict literal but invalid JSON (single quotes).
    with pytest.raises(ValueError, match="Invalid tool arguments JSON"):
        _parse(client, [_fake_tc("run_command", "{'command': 'ls'}")])


@pytest.mark.parametrize("provider,client", CLIENTS)
def test_arbitrary_code_never_executed(provider: str, client: object) -> None:
    sentinel = {"executed": False}

    class Exploit:
        def __repr__(self) -> str:
            sentinel["executed"] = True
            return "{}"

    # Simulate a payload whose evaluation would have side effects.
    # json.loads must reject it without executing anything.
    payload = "__import__('os').system('echo pwned')"
    with pytest.raises(ValueError, match="Invalid tool arguments JSON"):
        _parse(client, [_fake_tc("run_command", payload)])
    assert sentinel["executed"] is False


@pytest.mark.parametrize("provider,client", CLIENTS)
def test_none_and_passthrough_preserved(provider: str, client: object) -> None:
    assert _parse(client, None) == []
    parsed = _parse(client, [_fake_tc("echo", {"already": "dict"})])
    assert parsed[0].arguments == {"already": "dict"}
