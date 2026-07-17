"""PromptTuner — bộ khung giữ vòng lặp tối ưu prompt.

Đây là chỗ thể hiện Inversion of Control: framework nắm vòng lặp chính và
GỌI NGƯỢC lại 4 component do người dùng cắm vào.

    for mỗi vòng:
        ② executor.execute()    -> chạy prompt
        ③ evaluator.evaluate()  -> chấm điểm
        ① store.record_score()  -> lưu phiên bản + điểm
        ④ optimizer.propose()   -> đề xuất prompt mới
"""
import logging
from typing import List, Optional

from .interfaces import (BaseCallback, BaseEvaluator, BaseExecutor,
                         BaseOptimizer, BasePromptStore)
from .types import EvalResult, PromptVersion, Sample

logger = logging.getLogger(__name__)


class PromptTuner:
    """Bộ tinh chỉnh prompt tự động.

    :param executor:  ② thực thi prompt (bắt buộc)
    :param evaluator: ③ chấm điểm (bắt buộc)
    :param optimizer: ④ đề xuất prompt mới (bắt buộc)
    :param store:     ① kho prompt; mặc định dùng InMemoryPromptStore
    :param task_description: mô tả task, truyền cho optimizer
    :param max_iters: số vòng tối đa
    :param target_score: đạt điểm này thì dừng sớm (thang 0..100)
    :param patience: dừng nếu ngần này vòng liên tiếp không cải thiện
    :param callbacks: các hook theo dõi vòng lặp
    """

    def __init__(self,
                 executor: BaseExecutor,
                 evaluator: BaseEvaluator,
                 optimizer: BaseOptimizer,
                 store: Optional[BasePromptStore] = None,
                 task_description: str = "",
                 max_iters: int = 5,
                 target_score: float = 100.0,
                 patience: Optional[int] = None,
                 min_delta: float = 0.01,
                 callbacks: Optional[List[BaseCallback]] = None):
        if store is None:
            # import trễ để tránh vòng lặp import
            from ..components.stores.memory_store import InMemoryPromptStore
            store = InMemoryPromptStore()

        self.executor = executor
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.store = store
        self.task_description = task_description
        self.max_iters = max_iters
        self.target_score = target_score
        self.patience = patience
        self.min_delta = min_delta
        self.callbacks = callbacks or []

    # ---- hook helpers -------------------------------------------------
    def _fire(self, event: str, *args) -> None:
        for cb in self.callbacks:
            try:
                getattr(cb, event)(*args)
            except Exception:  # callback hỏng không được làm sập vòng lặp
                logger.exception("Callback %s lỗi", event)

    # ---- vòng lặp chính ----------------------------------------------
    def run(self, initial_prompt: str, samples: List[Sample]) -> Optional[PromptVersion]:
        """Chạy vòng lặp tối ưu, trả về phiên bản prompt tốt nhất."""
        if not samples:
            raise ValueError("Cần ít nhất 1 sample để đánh giá prompt.")

        self._fire("on_run_start", initial_prompt, samples)

        prompt = initial_prompt
        version = self.store.save(prompt, {"source": "initial"})
        best_so_far = float("-inf")
        stale = 0

        for i in range(self.max_iters):
            # ② Thực thi
            predictions = self.executor.execute(prompt, samples)
            # ③ Đánh giá
            result: EvalResult = self.evaluator.evaluate(prompt, predictions, samples)

            # Điểm chỉ tính trên ca chấm được -> quá nhiều ca lỗi là điểm bị thổi
            # phồng (9/12 ca lỗi + 3 đúng = 100/100). KHÔNG ghi nhận điểm đó,
            # cũng không để nó kích hoạt target_score, vì như vậy framework sẽ
            # tự tuyên bố tối ưu xong dựa trên số liệu rác.
            if not getattr(result, "reliable", True):
                logger.error(
                    "Vòng %s: chỉ chấm được %s/%s ca (%s ca lỗi) — điểm %.1f/100 "
                    "KHÔNG đáng tin nên không ghi nhận. Dừng lại. "
                    "Thường do rate limit/quota; hãy giảm num_workers hoặc chờ rồi chạy lại.",
                    i, result.num_scored, len(result.results), result.num_skipped,
                    result.score)
                self._fire("on_iteration_end", i, version, result)
                break

            # ① Quản lý prompt: ghi điểm cho phiên bản hiện tại
            self.store.record_score(version.version, result.score)
            version.score = result.score

            logger.info("Vòng %s: điểm %.1f/100 (%s/%s đúng)",
                        i, result.score, result.num_correct, len(result.results))
            self._fire("on_iteration_end", i, version, result)

            # Điều kiện dừng
            if result.score >= self.target_score:
                logger.info("Đạt target_score, dừng sớm.")
                break

            if result.score > best_so_far + self.min_delta:
                best_so_far = result.score
                stale = 0
            else:
                stale += 1
                if self.patience is not None and stale >= self.patience:
                    logger.info("Không cải thiện %s vòng, dừng.", stale)
                    break

            if i == self.max_iters - 1:
                break  # hết vòng, khỏi tốn 1 lần gọi optimizer

            # ④ Tối ưu: đề xuất prompt mới
            new_prompt = self.optimizer.propose(
                prompt, result,
                task_description=self.task_description,
                history=self.store.history(),
            )
            if not new_prompt or new_prompt.strip() == prompt.strip():
                logger.info("Optimizer không đề xuất được prompt mới, dừng.")
                break

            prompt = new_prompt
            version = self.store.save(prompt, {"source": "optimizer", "iteration": i + 1})

        best = self.store.best()
        self._fire("on_run_end", best)
        return best
