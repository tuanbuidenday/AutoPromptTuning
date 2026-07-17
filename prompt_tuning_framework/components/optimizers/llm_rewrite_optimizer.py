"""④ Optimizer tự viết: đưa lỗi cho một LLM mạnh và nhờ viết lại prompt.

Đây là optimizer "gốc" của framework — không phụ thuộc AutoPrompt.
"""
from typing import List, Optional

from ...core.interfaces import BaseOptimizer
from ...core.registry import register
from ...core.types import EvalResult, PromptVersion
from ...llm import build_llm, default_model

META_TEMPLATE = """You are an expert prompt engineer. Improve the prompt below.

## Task
{task_description}

## Current prompt (score: {score}/100)
{prompt}

## Previous attempts and their scores
{history}

## Cases where the current prompt FAILED
{errors}

Write a BETTER prompt that fixes these failures. Requirements:
- It must be a clear, self-contained instruction.
- It must be different from all previous prompts.
- It must make the model answer with exactly one of these labels: {labels}.
- Output ONLY the new prompt text, nothing else.
"""


@register("optimizer", "llm_rewrite")
class LLMRewriteOptimizer(BaseOptimizer):
    """:param model: LLM viết prompt mới — nên dùng model mạnh.
                     Để None thì lấy model mặc định của provider.
    """

    def __init__(self, model: Optional[str] = None, provider: str = "google",
                 labels: Optional[List[str]] = None, temperature: float = 0.8,
                 max_errors: int = 5, api_key: Optional[str] = None):
        self.model = model or default_model(provider, "optimizer")
        self.llm = build_llm(provider=provider, model=self.model,
                             temperature=temperature, api_key=api_key)
        self.labels = labels or []
        self.max_errors = max_errors

    def _format_errors(self, result: EvalResult) -> str:
        errs = result.errors[: self.max_errors]
        if not errs:
            return "(no failures — try making the prompt more precise)"
        return "\n".join(
            f"- Input: {e.sample.text}\n  Prompt answered: {e.predicted}\n  Correct answer: {e.expected}"
            for e in errs
        )

    def _format_history(self, history: Optional[List[PromptVersion]]) -> str:
        if not history:
            return "(none)"
        scored = [h for h in history if h.score is not None]
        return "\n".join(f"- (score {h.score}/100) {h.text}" for h in scored[-4:]) or "(none)"

    def propose(self, prompt: str, result: EvalResult, task_description: str = "",
                history: Optional[List[PromptVersion]] = None) -> str:
        msg = META_TEMPLATE.format(
            task_description=task_description or "(not provided)",
            prompt=prompt,
            score=result.score,
            history=self._format_history(history),
            errors=self._format_errors(result),
            labels=", ".join(self.labels) if self.labels else "the task's labels",
        )
        try:
            return (self.llm.invoke(msg).content or "").strip()
        except Exception:
            return ""  # thất bại -> tuner sẽ dừng an toàn
