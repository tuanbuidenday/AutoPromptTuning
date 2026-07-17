"""Kiểu dữ liệu dùng chung của framework."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Sample:
    """Một ca test: text đầu vào + nhãn đúng (nếu có)."""
    text: str
    label: Optional[str] = None
    id: Optional[int] = None


@dataclass
class Prediction:
    """Kết quả prompt đoán cho một ca test.

    :param model: model nào sinh ra dự đoán này. None = chạy một model duy nhất.
        Có trường này để đo prompt trên nhiều model: cùng một sample sẽ có nhiều
        Prediction, mỗi cái một model.
    """
    sample_id: int
    output: str
    model: Optional[str] = None


@dataclass
class SampleResult:
    """Đối chiếu 1 ca: prompt đoán gì vs đáp án, đúng hay sai.

    :param model: model đã sinh ra dự đoán (None nếu chạy một model).
    """
    sample: Sample
    predicted: str
    expected: Optional[str]
    correct: Optional[bool]
    model: Optional[str] = None


@dataclass
class EvalResult:
    """Kết quả đánh giá một prompt trên toàn bộ ca test.

    :param reliable: False khi có quá nhiều ca không chấm được (LLM lỗi/quota).
        Điểm chỉ tính trên các ca chấm được, nên nếu phần lớn ca bị lỗi thì điểm
        sẽ bị THỔI PHỒNG — 9/12 ca lỗi + 3 ca đúng = 100/100. Cờ này để tầng trên
        biết mà không tin vào điểm đó.
    """
    score: float                       # thang 0..100
    results: List[SampleResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reliable: bool = True
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """Khoảng tin cậy 95% (Wilson) của accuracy, thang 0..100.

        Luôn đọc con số này cạnh `score`. Với bộ mẫu nhỏ nó rất rộng: 16/16 ca
        đúng cho ra [80.6, 100.0] — tức "100 điểm" thật ra chỉ chứng minh được
        prompt đúng ÍT NHẤT ~81%. Báo cáo score mà giấu khoảng này là đang tạo
        cảm giác chắc chắn không có thật.
        """
        from .stats import wilson_interval
        return wilson_interval(self.num_correct, self.num_scored)

    @property
    def margin_of_error(self) -> float:
        """Nửa độ rộng khoảng tin cậy, tính bằng điểm. Càng lớn = càng ít tin được."""
        lo, hi = self.confidence_interval
        return (hi - lo) / 2

    def distinguishable_from(self, other: "EvalResult", alpha: float = 0.05) -> bool:
        """Điểm của hai prompt có khác nhau thật không, hay chỉ là nhiễu?

        Dùng kiểm định ghép cặp McNemar trên các ca bất đồng, nên CHỈ có nghĩa
        khi hai EvalResult chạy trên cùng một bộ mẫu, cùng thứ tự.

        Đây là câu hỏi mà `score` trần không trả lời nổi: 100.0 và 93.8 trên 16
        mẫu chỉ hơn nhau một ca, và một ca thì không chứng minh được gì.
        """
        from .stats import discordant_counts, mcnemar_exact
        a = [r.correct for r in self.results]
        b = [r.correct for r in other.results]
        only_a, only_b = discordant_counts(a, b)
        return mcnemar_exact(only_a, only_b) < alpha

    @property
    def errors(self) -> List[SampleResult]:
        """Các ca prompt đoán SAI — nguyên liệu để optimizer sửa prompt."""
        return [r for r in self.results if r.correct is False]

    @property
    def num_correct(self) -> int:
        return sum(1 for r in self.results if r.correct is True)

    @property
    def num_scored(self) -> int:
        """Số ca thực sự được chấm (mẫu số của điểm)."""
        return sum(1 for r in self.results if r.correct is not None)

    @property
    def num_skipped(self) -> int:
        """Số ca KHÔNG chấm được — thiếu nhãn hoặc gọi LLM lỗi."""
        return sum(1 for r in self.results if r.correct is None)


@dataclass
class PromptVersion:
    """Một phiên bản prompt trong kho (Registry)."""
    version: int
    text: str
    score: Optional[float] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
