"""4 điểm mở rộng (extension point) của framework.

Người dùng kế thừa các lớp này để cắm phần của mình vào. Framework (PromptTuner)
sẽ GỌI NGƯỢC lại chúng — đây chính là Inversion of Control.

Tương ứng 4 bước của vòng lặp khép kín:
    BasePromptStore -> Quản lý Prompt
    BaseExecutor    -> Thực thi
    BaseEvaluator   -> Đánh giá
    BaseOptimizer   -> Tối ưu hóa
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from .types import EvalResult, Prediction, PromptVersion, Sample


class BasePromptStore(ABC):
    """① Quản lý Prompt — kho lưu prompt kèm phiên bản và điểm."""

    @abstractmethod
    def save(self, prompt_text: str, metadata: Optional[dict] = None) -> PromptVersion:
        """Lưu một prompt mới, trả về phiên bản vừa tạo (tự tăng version)."""

    @abstractmethod
    def record_score(self, version: int, score: float) -> None:
        """Ghi điểm cho một phiên bản prompt."""

    @abstractmethod
    def history(self) -> List[PromptVersion]:
        """Toàn bộ phiên bản theo thứ tự tăng dần."""

    @abstractmethod
    def best(self) -> Optional[PromptVersion]:
        """Phiên bản có điểm cao nhất."""


class BaseExecutor(ABC):
    """② Thực thi — chạy prompt trên các ca test, trả về dự đoán."""

    @abstractmethod
    def execute(self, prompt: str, samples: List[Sample]) -> List[Prediction]:
        """Chạy `prompt` với từng sample, trả list Prediction."""


class BaseEvaluator(ABC):
    """③ Đánh giá — chấm điểm prompt dựa trên dự đoán vs đáp án."""

    @abstractmethod
    def evaluate(self, prompt: str, predictions: List[Prediction],
                 samples: List[Sample]) -> EvalResult:
        """Trả EvalResult với score thang 0..100 và chi tiết đúng/sai."""


class BaseOptimizer(ABC):
    """④ Tối ưu hóa — nhìn vào lỗi, đề xuất prompt tốt hơn."""

    @abstractmethod
    def propose(self, prompt: str, result: EvalResult,
                task_description: str = "",
                history: Optional[List[PromptVersion]] = None) -> str:
        """Trả về prompt mới. Trả chuỗi rỗng / prompt cũ nghĩa là 'bó tay'."""


class BaseCallback(ABC):
    """(Tùy chọn) Hook để theo dõi vòng lặp — logging, UI, biểu đồ..."""

    def on_run_start(self, prompt: str, samples: List[Sample]) -> None:
        pass

    def on_iteration_end(self, iteration: int, version: PromptVersion,
                         result: EvalResult) -> None:
        pass

    def on_run_end(self, best: Optional[PromptVersion]) -> None:
        pass
