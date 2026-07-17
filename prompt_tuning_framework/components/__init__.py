"""Các component sẵn có. Import ở đây để chúng tự đăng ký vào registry."""
from .evaluators import (AccuracyEvaluator, CompositeEvaluator,
                         CrossModelEvaluator)
from .executors import LLMExecutor, MultiModelExecutor
from .optimizers import LLMRewriteOptimizer
from .stores import InMemoryPromptStore, SQLitePromptStore

__all__ = [
    "InMemoryPromptStore", "SQLitePromptStore",
    "LLMExecutor", "MultiModelExecutor",
    "AccuracyEvaluator", "CompositeEvaluator", "CrossModelEvaluator",
    "LLMRewriteOptimizer",
]

# AutoPrompt là plugin tuỳ chọn (chỉ nạp khi repo AutoPrompt dùng được)
try:
    from .optimizers import AutoPromptOptimizer
    __all__.append("AutoPromptOptimizer")
except ImportError:  # pragma: no cover
    pass
