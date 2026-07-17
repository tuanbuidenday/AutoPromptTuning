"""③ Evaluator chấm điểm bằng độ chính xác (accuracy), thang 0..100."""
from typing import List

from ...core.interfaces import BaseEvaluator
from ...core.registry import register
from ...core.types import EvalResult, Prediction, Sample, SampleResult

ERROR_MARK = "__ERROR__"


@register("evaluator", "accuracy")
class AccuracyEvaluator(BaseEvaluator):
    """So khớp dự đoán với nhãn đúng (không phân biệt hoa/thường).

    :param case_sensitive: True nếu muốn so khớp phân biệt hoa/thường
    :param min_scored_ratio: tỉ lệ ca tối thiểu phải chấm được thì điểm mới đáng
        tin. Điểm = đúng/chấm-được, nên ca lỗi bị loại khỏi MẪU SỐ: 9/12 ca lỗi
        + 3 ca đúng = 100/100 dù prompt rất dở. Dưới ngưỡng này thì đánh dấu
        reliable=False để tuner không tin vào điểm.
    """

    def __init__(self, case_sensitive: bool = False, min_scored_ratio: float = 0.8):
        self.case_sensitive = case_sensitive
        self.min_scored_ratio = min_scored_ratio

    def _match(self, pred: str, gold: str) -> bool:
        if self.case_sensitive:
            return pred.strip() == gold.strip()
        return pred.strip().lower() == gold.strip().lower()

    def evaluate(self, prompt: str, predictions: List[Prediction],
                 samples: List[Sample]) -> EvalResult:
        by_id = {p.sample_id: p.output for p in predictions}
        results: List[SampleResult] = []
        n_ok = n_scored = 0

        for s in samples:
            pred = by_id.get(s.id, "")
            if s.label is None or pred.startswith(ERROR_MARK):
                correct = None  # không chấm: thiếu đáp án hoặc gọi LLM lỗi
            else:
                correct = self._match(pred, s.label)
                n_scored += 1
                n_ok += int(correct)
            results.append(SampleResult(sample=s, predicted=pred,
                                        expected=s.label, correct=correct))

        score = round(n_ok / n_scored * 100, 1) if n_scored else 0.0
        reliable = bool(samples) and n_scored >= self.min_scored_ratio * len(samples)
        return EvalResult(
            score=score,
            results=results,
            reliable=reliable,
            metadata={"num_scored": n_scored, "num_correct": n_ok,
                      "num_skipped": len(samples) - n_scored},
        )
