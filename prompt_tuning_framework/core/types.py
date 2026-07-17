"""Kiểu dữ liệu dùng chung của framework."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Sample:
    """Một ca test: text đầu vào + nhãn đúng (nếu có)."""
    text: str
    label: Optional[str] = None
    id: Optional[int] = None


@dataclass
class Prediction:
    """Kết quả prompt đoán cho một ca test."""
    sample_id: int
    output: str


@dataclass
class SampleResult:
    """Đối chiếu 1 ca: prompt đoán gì vs đáp án, đúng hay sai."""
    sample: Sample
    predicted: str
    expected: Optional[str]
    correct: Optional[bool]


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
