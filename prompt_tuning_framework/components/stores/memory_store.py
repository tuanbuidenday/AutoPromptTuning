"""① PromptStore lưu trong bộ nhớ — mặc định, không cần DB."""
from datetime import datetime
from typing import List, Optional

from ...core.interfaces import BasePromptStore
from ...core.registry import register
from ...core.types import PromptVersion


@register("store", "memory")
class InMemoryPromptStore(BasePromptStore):
    def __init__(self):
        self._versions: List[PromptVersion] = []

    def save(self, prompt_text: str, metadata: Optional[dict] = None) -> PromptVersion:
        pv = PromptVersion(
            version=len(self._versions),
            text=prompt_text,
            created_at=datetime.now().isoformat(timespec="seconds"),
            metadata=metadata or {},
        )
        self._versions.append(pv)
        return pv

    def record_score(self, version: int, score: float) -> None:
        for pv in self._versions:
            if pv.version == version:
                pv.score = score
                return

    def history(self) -> List[PromptVersion]:
        return list(self._versions)

    def best(self) -> Optional[PromptVersion]:
        scored = [v for v in self._versions if v.score is not None]
        return max(scored, key=lambda v: v.score) if scored else None
