"""③ Evaluator đa mục tiêu: vừa đúng vừa ngắn.

Bài toán "rút gọn prompt mà không mất độ chính xác" có HAI mục tiêu chống nhau.
`EvalResult.score` chỉ là một float, nên buộc phải quy về một số duy nhất
(scalarize) để tuner so sánh được. Việc gộp đó luôn làm mất thông tin, nên ta
vẫn giữ đủ từng thành phần trong `metrics` để báo cáo trung thực.
"""
from typing import List, Optional

from ...core.interfaces import BaseEvaluator
from ...core.registry import register
from ...core.types import EvalResult, Prediction, Sample


def count_words(text: str) -> int:
    """Đếm từ. Cố tình KHÔNG dùng tokenizer của model nào.

    Mỗi model tokenize một kiểu, nên số token của model A không so được với model
    B. Framework này nhắm tới prompt dùng chung cho nhiều model, nên cần một
    thước đo trung lập, ổn định và không phải cài thêm thư viện. Đếm từ là xấp xỉ
    thô nhưng nhất quán — đủ để so prompt này với prompt kia.
    """
    return len(text.split())


@register("evaluator", "composite")
class CompositeEvaluator(BaseEvaluator):
    """Gộp độ chính xác với độ ngắn gọn thành một điểm duy nhất.

        điểm = accuracy - brevity_weight * phần_trăm_vượt_ngân_sách_từ

    Chỉ phạt khi prompt DÀI HƠN `word_budget`; ngắn hơn không được thưởng thêm.
    Cố ý như vậy: nếu thưởng cho việc ngắn thì optimizer sẽ đua nhau cắt prompt
    tới mức cụt lủn, đánh đổi cả độ chính xác để lấy điểm ngắn.

    :param base: evaluator đo độ chính xác (mặc định AccuracyEvaluator)
    :param word_budget: số từ coi là "vừa đủ ngắn"; vượt mới bị trừ
    :param brevity_weight: trừ bao nhiêu điểm cho mỗi 100% vượt ngân sách.
        Mặc định thấp (10) vì độ chính xác quan trọng hơn độ ngắn — prompt ngắn
        mà sai thì vô dụng.
    """

    def __init__(self, base: Optional[BaseEvaluator] = None,
                 word_budget: int = 50,
                 brevity_weight: float = 10.0,
                 case_sensitive: bool = False,
                 min_scored_ratio: float = 0.8):
        if word_budget <= 0:
            raise ValueError(f"word_budget phải > 0, nhận {word_budget}")
        if brevity_weight < 0:
            raise ValueError(f"brevity_weight không được âm, nhận {brevity_weight}")

        if base is None:
            from .accuracy_evaluator import AccuracyEvaluator
            base = AccuracyEvaluator(case_sensitive=case_sensitive,
                                     min_scored_ratio=min_scored_ratio)
        self.base = base
        self.word_budget = word_budget
        self.brevity_weight = brevity_weight

    def evaluate(self, prompt: str, predictions: List[Prediction],
                 samples: List[Sample]) -> EvalResult:
        result = self.base.evaluate(prompt, predictions, samples)

        words = count_words(prompt)
        over = max(0.0, (words - self.word_budget) / self.word_budget)
        penalty = self.brevity_weight * over
        combined = max(0.0, result.score - penalty)

        result.metrics.update({
            "accuracy": result.score,
            "prompt_words": float(words),
            "prompt_chars": float(len(prompt)),
            "brevity_penalty": round(penalty, 2),
        })
        result.metadata.update({
            "accuracy_truoc_khi_phat": result.score,
            "word_budget": self.word_budget,
        })
        result.score = round(combined, 1)
        return result
