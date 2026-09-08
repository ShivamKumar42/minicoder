from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Tool(ABC):
    name: str = ""
    description: str = ""
    _parameter_schema: Optional[Dict[str, Any]] = None

    @property
    def parameter_schema(self) -> Dict[str, Any]:
        if self._parameter_schema is None:
            self._parameter_schema = self._build_schema()
        return self._parameter_schema

    @abstractmethod
    def _build_schema(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        pass