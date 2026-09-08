from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Type, ClassVar


def _load_env() -> None:
    """Load environment variables from .env file if present."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(env_path, override=False)


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
    
    # Workspace
    workspace: str = field(default=".")
    
    def __post_init__(self) -> None:
        _load_env()
        self._initialize_from_env()
    
    def _initialize_from_env(self) -> None:
        """Initialize settings from environment variables."""
        # Provider
        env_provider = os.environ.get("MINICODER_PROVIDER")
        if env_provider:
            self.provider = env_provider
        
        # Model
        env_model = os.environ.get("MINICODER_MODEL")
        if env_model:
            self.model = env_model
        
        # Base URL (for openai-compatible)
        env_base_url = os.environ.get("MINICODER_BASE_URL")
        if env_base_url:
            self.base_url = env_base_url
        
        # API key
        env_api_key = os.environ.get("MINICODER_API_KEY")
        if env_api_key:
            self.api_key = env_api_key
        
        # Iterations
        env_max_iters = os.environ.get("MINICODER_MAX_ITERATIONS")
        if env_max_iters:
            try:
                self.max_iterations = int(env_max_iters)
            except ValueError:
                pass
        
        # Tool calls limit
        env_max_tool = os.environ.get("MINICODER_MAX_TOOL_CALLS")
        if env_max_tool:
            try:
                self.max_tool_calls = int(env_max_tool)
            except ValueError:
                pass
        
        # Timeout
        env_timeout = os.environ.get("MINICODER_TIMEOUT")
        if env_timeout:
            try:
                self.timeout = int(env_timeout)
            except ValueError:
                pass
        
        # Dry run
        env_dry = os.environ.get("MINICODER_DRY_RUN")
        if env_dry:
            self.dry_run = env_dry.lower() in ("true", "1", "yes")
        
        # Approval mode
        env_approval = os.environ.get("MINICODER_APPROVAL_MODE")
        if env_approval:
            self.approval_mode = env_approval
        
        # Verbose
        env_verbose = os.environ.get("MINICODER_VERBOSE")
        if env_verbose:
            self.verbose = env_verbose.lower() in ("true", "1", "yes")
        
        # Workspace
        env_workspace = os.environ.get("MINICODER_WORKSPACE")
        if env_workspace:
            self.workspace = env_workspace
    
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