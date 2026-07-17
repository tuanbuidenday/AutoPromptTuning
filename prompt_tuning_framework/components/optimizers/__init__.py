from .llm_rewrite_optimizer import LLMRewriteOptimizer

__all__ = ["LLMRewriteOptimizer"]

# AutoPrompt là plugin tuỳ chọn: chỉ nạp được khi repo AutoPrompt sẵn sàng
try:
    from .autoprompt_optimizer import AutoPromptOptimizer
    __all__.append("AutoPromptOptimizer")
except Exception:  # pragma: no cover
    pass
