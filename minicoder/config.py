from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Dict, Any


def _load_env() -> None:
    """Load environment variables from .env file if present."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(env_path, override=False)


# Environment variable mappings: env name -> (setting field, kind).
# Kinds: "str" (override when set and non-empty), "int" (override when
# parseable), "positive_int" (override when a positive integer),
# "bool" (true for "true"/"1"/"yes", case-insensitive).
_ENV_VARS: tuple[tuple[str, str, str], ...] = (
    ("MINICODER_PROVIDER", "provider", "str"),
    ("MINICODER_MODEL", "model", "str"),
    ("MINICODER_BASE_URL", "base_url", "str"),
    ("MINICODER_API_KEY", "api_key", "str"),
    ("MINICODER_APPROVAL_MODE", "approval_mode", "str"),
    ("MINICODER_WORKSPACE", "workspace", "str"),
    ("MINICODER_MAX_ITERATIONS", "max_iterations", "int"),
    ("MINICODER_MAX_TOOL_CALLS", "max_tool_calls", "int"),
    ("MINICODER_TIMEOUT", "timeout", "positive_int"),
    ("MINICODER_MAX_TOOL_OUTPUT_CHARS", "max_tool_output_chars", "positive_int"),
    ("MINICODER_MAX_HISTORY", "max_history", "positive_int"),
    ("MINICODER_DRY_RUN", "dry_run", "bool"),
    ("MINICODER_VERBOSE", "verbose", "bool"),
)


@dataclass
class Settings:
    """Configuration settings for MiniCoder."""
    
    # Core provider settings
    provider: str = field(default="openai")
    model: str = field(default="gpt-4o-mini")
    base_url: str = field(default="")
    api_key: str = field(default="")
    
    # Runtime settings
    max_iterations: int = field(default=100)
    max_tool_calls: int = field(default=200)
    timeout: int = field(default=60)
    dry_run: bool = field(default=False)
    approval_mode: str = field(default="auto")
    verbose: bool = field(default=False)

    # Context-budget: max characters kept per tool result. Larger outputs
    # are head-truncated with an omission note. See tools/budget.py.
    max_tool_output_chars: int = field(default=8000)

    # History bound: at most this many messages are kept in agent state
    # (the seed message is always preserved).
    max_history: int = field(default=100)
    
    # Workspace
    workspace: str = field(default=".")
    
    def __post_init__(self) -> None:
        _load_env()
        self._initialize_from_env()
    
    def _initialize_from_env(self) -> None:
        """Initialize settings from environment variables."""
        for env_name, field_name, kind in _ENV_VARS:
            raw = os.environ.get(env_name)
            if not raw:
                continue
            if kind == "str":
                setattr(self, field_name, raw)
            elif kind == "bool":
                setattr(self, field_name, raw.lower() in ("true", "1", "yes"))
            else:  # "int" or "positive_int"
                try:
                    value = int(raw)
                except ValueError:
                    continue
                if kind == "positive_int" and value <= 0:
                    continue
                setattr(self, field_name, value)
    
    def get_provider_config(self, key: str, default: str = "") -> str:
        """Get provider-specific configuration."""
        config_key = f"MINICODER_{self.provider.upper}_{key}"
        env_val = os.environ.get(config_key)
        if env_val is not None:
            return env_val
        return default
    
    def override_with_provider_config(self, overrides: Dict[str, Any]) -> None:
        """Override settings with provider-specific configuration."""
        for key, value in overrides.items():
            setattr(self, key, value)


# Provider-specific environment variable mappings
PROVIDER_ENV_VARS: Dict[str, Dict[str, str]] = {
    "openai": {
        "api_key": "OPENAI_API_KEY",
        "model": "OPENAI_MODEL",
        "base_url": "OPENAI_BASE_URL",
    },
    "anthropic": {
        "api_key": "ANTHROPIC_API_KEY",
        "model": "ANTHROPIC_MODEL",
        "base_url": "",  # Anthropic doesn't use base_url
    },
    "openai-compatible": {
        "api_key": "OPENAI_COMPATIBLE_API_KEY",
        "model": "OPENAI_COMPATIBLE_MODEL",
        "base_url": "OPENAI_COMPATIBLE_BASE_URL",
    },
}


def get_settings() -> Settings:
    """Get the global settings instance."""
    return Settings()