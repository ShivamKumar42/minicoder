from __future__ import annotations
from typing import Any, ClassVar, List, Optional, Dict

from .openai_base import OpenAIChatBase


class OpenAIClient(OpenAIChatBase):
    _PROVIDER_NAME: ClassVar[str] = "openai"
    _ENV_API_KEY: ClassVar[str] = "OPENAI_API_KEY"
    _ENV_BASE_URL: ClassVar[Optional[str]] = None
    _DEFAULT_BASE_URL: ClassVar[Optional[str]] = None
