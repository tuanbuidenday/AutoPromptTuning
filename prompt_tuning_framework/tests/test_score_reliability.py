"""Test chống THỔI PHỒNG ĐIỂM khi gọi LLM bị lỗi.

Bug thật đã xảy ra: điểm = đúng / chấm-được, nên ca lỗi bị loại khỏi MẪU SỐ.
Chạy ví dụ khó với Gemini bị dính rate limit 429 -> 9/12 ca lỗi, 3 ca còn lại
đúng hết -> framework báo 100/100 cho một prompt thực chất chỉ 50/100, rồi
DỪNG SỚM vì tưởng đã đạt target_score.
"""
import pytest

from prompt_tuning_framework import PromptTuner, Sample
from prompt_tuning_framework.components import AccuracyEvaluator
from prompt_tuning_framework.core.interfaces import BaseExecutor, BaseOptimizer
from prompt_tuning_framework.core.types import Prediction

LABELS = ["Yes", "No"]


def _samples(n=12):
    return [Sample(id=i, text=f"ca {i}", label="Yes" if i < n // 2 else "No")
            for i in range(n)]


class FlakyNetworkExecutor(BaseExecutor):
    """Đoán bừa 'Yes'; `n_loi` ca đầu trả về lỗi giống hệt quota 429."""

    def __init__(self, n_loi):
        self.n_loi = n_loi

    def execute(self, prompt, samples):
        return [
            Prediction(sample_id=s.id,
                       output="__ERROR__: 429 quota exceeded" if i < self.n_loi else "Yes")
            for i, s in enumerate(samples)
        ]


class SilentOptimizer(BaseOptimizer):
    def propose(self, prompt, result, task_description="", history=None):
        return prompt + " v2"


# ---------------- Evaluator ----------------

def test_no_errors_scores_everything_correctly():
    preds = [Prediction(sample_id=i, output="Yes") for i in range(12)]
    r = AccuracyEvaluator().evaluate("p", preds, _samples())
    assert r.score == 50.0          # 6/12 đúng
    assert r.num_scored == 12 and r.num_skipped == 0
    assert r.reliable is True


def test_too_many_errors_marks_result_unreliable():
    """Đúng kịch bản đã xảy ra: 9/12 ca lỗi, 3 ca đúng -> điểm 100 nhưng phải cờ đỏ."""
    preds = [Prediction(sample_id=i,
                        output="Yes" if i < 3 else "__ERROR__: 429 quota")
             for i in range(12)]
    r = AccuracyEvaluator().evaluate("p", preds, _samples())
    assert r.score == 100.0, "điểm vẫn bị thổi phồng — đó là lý do cần cờ reliable"
    assert r.num_scored == 3 and r.num_skipped == 9
    assert r.reliable is False, "phải bị đánh dấu KHÔNG đáng tin"


def test_few_errors_stays_reliable():
    """1/12 ca lỗi (>= 80% chấm được) -> vẫn tin được."""
    preds = [Prediction(sample_id=i, output="__ERROR__: x" if i == 0 else "Yes")
             for i in range(12)]
    r = AccuracyEvaluator().evaluate("p", preds, _samples())
    assert r.num_scored == 11
    assert r.reliable is True


def test_min_scored_ratio_threshold_is_configurable():
    preds = [Prediction(sample_id=i, output="__ERROR__: x" if i < 5 else "Yes")
             for i in range(12)]
    samples = _samples()
    assert AccuracyEvaluator(min_scored_ratio=0.8).evaluate("p", preds, samples).reliable is False
    assert AccuracyEvaluator(min_scored_ratio=0.5).evaluate("p", preds, samples).reliable is True


def test_all_errored_scores_0_and_is_unreliable():
    preds = [Prediction(sample_id=i, output="__ERROR__: x") for i in range(12)]
    r = AccuracyEvaluator().evaluate("p", preds, _samples())
    assert r.score == 0.0 and r.num_scored == 0
    assert r.reliable is False


# ---------------- Tuner ----------------

def test_tuner_does_not_record_an_inflated_score():
    """Điểm 100 giả KHÔNG được lưu vào store, nên không thể thành 'tốt nhất'."""
    tuner = PromptTuner(
        executor=FlakyNetworkExecutor(n_loi=9),
        evaluator=AccuracyEvaluator(),
        optimizer=SilentOptimizer(),
        max_iters=3,
        target_score=100.0,
    )
    best = tuner.run("prompt do", _samples())

    scores = [v.score for v in tuner.store.history()]
    assert scores == [None], f"điểm giả đã lọt vào store: {scores}"
    assert best is None, "không được tuyên bố có prompt tốt nhất từ số liệu rác"


def test_tuner_does_not_stop_early_on_a_fake_score():
    """Điểm 100 giả không được kích hoạt target_score như một chiến thắng."""
    tuner = PromptTuner(
        executor=FlakyNetworkExecutor(n_loi=9),
        evaluator=AccuracyEvaluator(),
        optimizer=SilentOptimizer(),
        max_iters=3,
        target_score=100.0,
    )
    tuner.run("prompt do", _samples())
    # Chỉ có prompt khởi tạo — dừng ngay khi phát hiện không đáng tin,
    # không đi tiếp để đốt thêm quota vào số liệu rác.
    assert len(tuner.store.history()) == 1


def test_tuner_errors_clearly(caplog):
    tuner = PromptTuner(
        executor=FlakyNetworkExecutor(n_loi=9),
        evaluator=AccuracyEvaluator(),
        optimizer=SilentOptimizer(),
        max_iters=2,
    )
    with caplog.at_level("ERROR"):
        tuner.run("prompt do", _samples())
    assert "KHÔNG đáng tin" in caplog.text
    assert "3/12" in caplog.text


def test_tuner_runs_normally_when_there_are_no_errors():
    """Không được làm hỏng luồng bình thường."""
    tuner = PromptTuner(
        executor=FlakyNetworkExecutor(n_loi=0),
        evaluator=AccuracyEvaluator(),
        optimizer=SilentOptimizer(),
        max_iters=2,
    )
    best = tuner.run("prompt do", _samples())
    assert best is not None and best.score == 50.0
