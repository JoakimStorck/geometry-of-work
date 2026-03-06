from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Tuple

@dataclass
class SessionCache:
    """Simple in-memory cache for a Python session."""
    data: Dict[Hashable, Any] = field(default_factory=dict)

    def get(self, key: Hashable) -> Any:
        return self.data.get(key)

    def set(self, key: Hashable, value: Any) -> Any:
        self.data[key] = value
        return value

SESSION_CACHE = SessionCache()
