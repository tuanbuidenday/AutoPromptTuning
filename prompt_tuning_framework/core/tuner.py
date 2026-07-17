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
    def run(self, initial_prompt: str, samples: List[Sample],
            test_samples: Optional[List[Sample]] = None) -> Optional[PromptVersion]:
        """Chạy vòng lặp tối ưu, trả về phiên bản prompt tốt nhất.

        :param samples: tập DEV — dùng để chấm và để optimizer xem lỗi mà sửa prompt.
        :param test_samples: tập TEST giữ riêng, optimizer KHÔNG bao giờ thấy. Nếu
            truyền vào, prompt tốt nhất sẽ được chấm thêm một lần trên tập này sau
            khi vòng lặp kết thúc, và kết quả ghi vào ``best.metadata``.

            Nên truyền. Điểm trên tập dev là điểm HỌC THUỘC: optimizer đã được xem
            đúng những ca sai đó rồi viết prompt vá chúng, nên đạt 100/100 trên dev
            là chuyện hiển nhiên và không chứng minh prompt tốt với ca mới. Chỉ có
            điểm trên tập test mới nói lên khả năng khái quát hoá.
        """
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
        if best is not None and test_samples:
            self._evaluate_on_test(best, test_samples)
        self._fire("on_run_end", best)
        return best

    # ---- tập test giữ riêng -------------------------------------------
    def _evaluate_on_test(self, best: PromptVersion,
                          test_samples: List[Sample]) -> None:
        """Chấm prompt tốt nhất trên tập test và ghi kết quả vào metadata.

        Chỉ chạy MỘT lần, sau khi vòng lặp đã chốt. Nếu dùng điểm test để chọn
        prompt thì tập test lập tức biến thành tập dev thứ hai và mất hết ý nghĩa.
        """
        predictions = self.executor.execute(best.text, test_samples)
        result = self.evaluator.evaluate(best.text, predictions, test_samples)

        lo, hi = result.confidence_interval
        best.metadata.update({
            "test_score": result.score,
            "test_ci_low": round(lo, 1),
            "test_ci_high": round(hi, 1),
            "test_num_scored": result.num_scored,
            "test_reliable": result.reliable,
        })

        if not result.reliable:
            logger.error(
                "Tập test: chỉ chấm được %s/%s ca — điểm %.1f/100 KHÔNG đáng tin.",
                result.num_scored, len(test_samples), result.score)
            return

        logger.info("Tập test (giữ riêng): %.1f/100 — khoảng tin cậy 95%% [%.1f, %.1f] "
                    "trên %s ca.", result.score, lo, hi, result.num_scored)

        # Điểm dev cao hơn hẳn điểm test = prompt đã vá thuộc lòng các ca dev.
        # Đây là dấu hiệu học thuộc, và là lý do tồn tại của tập test.
        dev_score = best.score
        if dev_score is not None:
            gap = dev_score - result.score
            if gap > 10:
                logger.warning(
                    "HỌC THUỘC: điểm dev %.1f nhưng test chỉ %.1f (chênh %.1f điểm). "
                    "Prompt đang vá riêng các ca dev thay vì học quy luật chung. "
                    "Hãy báo cáo điểm TEST, đừng báo cáo điểm dev.",
                    dev_score, result.score, gap)
