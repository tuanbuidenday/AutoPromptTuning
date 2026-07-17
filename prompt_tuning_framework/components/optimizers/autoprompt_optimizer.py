"""④ Optimizer adapter: dùng meta-chain của AutoPrompt làm engine tối ưu.

Đây là ví dụ cho thấy framework có thể "cắm" một engine bên ngoài vào mà không
phải sửa lõi. AutoPrompt chỉ là MỘT trong các optimizer, không phải bắt buộc.
"""
import json
import sys
from pathlib import Path
from typing import List, Optional

from ...core.interfaces import BaseOptimizer
from ...core.registry import register
from ...core.types import EvalResult, PromptVersion

# AutoPrompt nằm ở gốc repo
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_META_FOLDER = "prompts/meta_prompts_classification"


@register("optimizer", "autoprompt")
class AutoPromptOptimizer(BaseOptimizer):
    """Gọi chain `step_prompt` của AutoPrompt để đề xuất prompt mới.

    :param model / provider: LLM dùng cho meta-prompt (model=None -> mặc định của provider)
    :param labels: label schema, AutoPrompt yêu cầu
    :param meta_folder: thư mục meta-prompt của AutoPrompt
    """

    def __init__(self, model: Optional[str] = None, provider: str = "google",
                 labels: Optional[List[str]] = None, temperature: float = 0.8,
                 meta_folder: str = DEFAULT_META_FOLDER, max_errors: int = 4):
        from ...llm import default_model
        model = model or default_model(provider, "optimizer")
        self.model = model

        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))

        from easydict import EasyDict  # noqa: E402
        from utils.llm_chain import ChainWrapper, get_chain_metadata  # noqa: E402

        self.labels = labels or []
        self.max_errors = max_errors
        prompt_path = _REPO_ROOT / meta_folder / "step_prompt.prompt"
        meta = get_chain_metadata(prompt_path)
        # AutoPrompt truy cập config bằng thuộc tính (llm_config.type) -> cần EasyDict,
        # và so sánh type == 'google' (chữ thường) nên phải lowercase.
        llm_config = EasyDict({"type": provider.lower(), "name": model,
                               "temperature": temperature})
        self.chain = ChainWrapper(llm_config, prompt_path,
                                  meta["json_schema"], meta["parser_func"])
        self._provider = provider.lower()

    def _format_history(self, history: Optional[List[PromptVersion]],
                        result: EvalResult) -> str:
        rows = []
        for h in (history or []):
            if h.score is not None:
                rows.append(f"####\n##Prompt Score: {h.score:.2f}\n##Prompt:\n{h.text}\n#################")
        return "\n".join(rows[-4:]) or "(none)"

    def _format_error_analysis(self, result: EvalResult) -> str:
        errs = result.errors[: self.max_errors]
        if not errs:
            return "The prompt made no mistakes on the current benchmark; make it more precise and robust."
        lines = [f"The prompt scored {result.score}/100. Failure cases:"]
        for e in errs:
            lines.append(f"- Input: {e.sample.text}\n  Predicted: {e.predicted} | Correct: {e.expected}")
        lines.append("Modify the prompt so these cases are classified correctly.")
        return "\n".join(lines)

    def propose(self, prompt: str, result: EvalResult, task_description: str = "",
                history: Optional[List[PromptVersion]] = None) -> str:
        chain_input = {
            "task_description": task_description or "(not provided)",
            "history": self._format_history(history, result),
            "error_analysis": self._format_error_analysis(result),
            "labels": json.dumps(self.labels),
        }
        suggestion = self.chain.invoke(chain_input)
        # Gemini trả tool-call dạng list -> bóc 'args' (quirk của AutoPrompt)
        if self._provider == "google" and isinstance(suggestion, list) and len(suggestion) == 1:
            suggestion = suggestion[0]["args"]
        if not suggestion or "prompt" not in suggestion:
            return ""
        return str(suggestion["prompt"]).strip()
