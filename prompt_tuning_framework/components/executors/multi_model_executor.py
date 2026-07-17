"""② Executor chạy CÙNG một prompt trên NHIỀU model.

Để trả lời câu "prompt này có tốt cho nhiều model không?" — một prompt chỉ chạy
tốt trên đúng model đã tinh chỉnh nó thì không dùng lại được ở đâu khác.
"""
from typing import Any, Dict, List, Optional

from ...core.interfaces import BaseExecutor
from ...core.registry import register
from ...core.types import Prediction, Sample
from .llm_executor import DEFAULT_TEMPLATE, LLMExecutor


@register("executor", "multi_model")
class MultiModelExecutor(BaseExecutor):
    """Chạy prompt trên nhiều model, gắn nhãn model vào từng dự đoán.

    Trả về len(models) × len(samples) dự đoán. Ghép với ``CrossModelEvaluator``
    để chấm theo model yếu nhất.

    Cảnh báo chi phí: số request nhân lên theo số model. Với free tier Gemini
    (20 request/ngày/model) thì 3 model không có nghĩa là hết quota nhanh gấp 3 —
    quota tính RIÊNG cho từng model, nên chia tải qua nhiều model thực ra lại
    giúp chạy được nhiều hơn.

    :param models: danh sách dict, mỗi dict là tham số cho một LLMExecutor, vd
        ``[{"provider": "google", "model": "gemini-3.1-flash-lite"},
           {"provider": "openai", "model": "gpt-4o-mini"}]``
    """

    def __init__(self, models: List[Dict[str, Any]],
                 labels: Optional[List[str]] = None,
                 temperature: float = 0.0, num_workers: int = 1,
                 template: str = DEFAULT_TEMPLATE, delay: float = 0.0):
        if not models:
            raise ValueError("Cần ít nhất 1 model trong 'models'.")

        self.executors: List[LLMExecutor] = []
        for spec in models:
            kwargs = dict(spec)
            # Tham số chung; từng model vẫn ghi đè được bằng chính spec của nó.
            kwargs.setdefault("labels", labels)
            kwargs.setdefault("temperature", temperature)
            kwargs.setdefault("num_workers", num_workers)
            kwargs.setdefault("template", template)
            kwargs.setdefault("delay", delay)
            self.executors.append(LLMExecutor(**kwargs))

        names = [e.model for e in self.executors]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Tên model bị trùng: {names}. Kết quả gộp theo tên model nên "
                f"trùng tên sẽ khiến các model đè lên nhau.")

    @property
    def models(self) -> List[str]:
        return [e.model for e in self.executors]

    def execute(self, prompt: str, samples: List[Sample]) -> List[Prediction]:
        out: List[Prediction] = []
        for ex in self.executors:
            out.extend(ex.execute(prompt, samples))
        return out
