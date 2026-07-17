"""③ Evaluator đo prompt trên nhiều model, chấm theo model YẾU NHẤT."""
from collections import defaultdict
from typing import Dict, List, Optional

from ...core.interfaces import BaseEvaluator
from ...core.registry import register
from ...core.types import EvalResult, Prediction, Sample


@register("evaluator", "cross_model")
class CrossModelEvaluator(BaseEvaluator):
    """Chấm prompt trên từng model rồi lấy điểm của model TỆ NHẤT.

    Vì sao lấy min chứ không lấy trung bình: trung bình cho phép một model giỏi
    che lấp một model dở. Prompt đạt 100 trên model A và 60 trên model B có trung
    bình 80 — nghe ổn, nhưng nó KHÔNG phải prompt dùng được cho nhiều model.
    Lấy min buộc optimizer phải sửa cho model yếu nhất, và đó đúng là định nghĩa
    của "chạy tốt trên nhiều model".

    Kết quả trả về là EvalResult CỦA MODEL TỆ NHẤT, không phải gộp tất cả:

    - ``confidence_interval`` giữ được tính đúng đắn. Nếu gộp N model × M mẫu
      thành một mẫu số N*M thì khoảng tin cậy sẽ hẹp đi giả tạo — các quan sát đó
      không độc lập, chúng là cùng M mẫu được đo lặp lại.
    - ``errors`` là lỗi của model yếu nhất, nên optimizer sửa đúng chỗ đang hỏng.

    Điểm từng model vẫn được giữ đủ trong ``metrics`` để báo cáo.

    :param base: evaluator chấm cho từng model (mặc định AccuracyEvaluator)
    """

    def __init__(self, base: Optional[BaseEvaluator] = None,
                 case_sensitive: bool = False, min_scored_ratio: float = 0.8):
        if base is None:
            from .accuracy_evaluator import AccuracyEvaluator
            base = AccuracyEvaluator(case_sensitive=case_sensitive,
                                     min_scored_ratio=min_scored_ratio)
        self.base = base

    def evaluate(self, prompt: str, predictions: List[Prediction],
                 samples: List[Sample]) -> EvalResult:
        by_model: Dict[Optional[str], List[Prediction]] = defaultdict(list)
        for p in predictions:
            by_model[p.model].append(p)

        # Một model (hoặc dự đoán không gắn nhãn model) -> hành xử y như thường.
        if len(by_model) <= 1:
            return self.base.evaluate(prompt, predictions, samples)

        per_model: Dict[str, EvalResult] = {}
        for model, preds in by_model.items():
            r = self.base.evaluate(prompt, preds, samples)
            for sr in r.results:
                sr.model = model
            per_model[str(model)] = r

        worst_name = min(per_model, key=lambda m: per_model[m].score)
        worst = per_model[worst_name]

        scores = {m: r.score for m, r in per_model.items()}
        worst_score = min(scores.values())
        best_score = max(scores.values())

        worst.metrics.update({f"accuracy__{m}": s for m, s in scores.items()})
        worst.metrics.update({
            "accuracy_min": worst_score,
            "accuracy_max": best_score,
            "accuracy_mean": round(sum(scores.values()) / len(scores), 1),
            # Chênh lệch lớn = prompt kén model. Prompt dùng chung tốt thì con số
            # này phải nhỏ, chứ không chỉ cần accuracy_min cao.
            "accuracy_spread": round(best_score - worst_score, 1),
        })
        worst.metadata.update({
            "models": sorted(per_model),
            "worst_model": worst_name,
            "per_model_score": scores,
        })
        # Một model chấm không đủ ca thì cả kết quả không đáng tin: model đó có
        # thể đang "được" điểm cao chỉ vì phần lớn ca của nó lỗi và bị loại khỏi
        # mẫu số — đúng cái bẫy thổi phồng điểm mà cờ reliable sinh ra để chặn.
        worst.reliable = all(r.reliable for r in per_model.values())
        return worst
