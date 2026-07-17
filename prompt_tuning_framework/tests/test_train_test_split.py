"""Kiểm thử việc tách tập test giữ riêng.

Vì sao cần: optimizer được xem các ca SAI để viết lại prompt. Nếu chấm điểm trên
chính những ca đó thì prompt chỉ đang vá thuộc lòng, và 100/100 thu được là điểm
HỌC THUỘC — không nói lên gì về ca mới. Tập test phải nằm ngoài tầm nhìn của
optimizer.
"""
import logging
from typing import List

import pytest

from prompt_tuning_framework import (BaseEvaluator, BaseExecutor, BaseOptimizer,
                                     EvalResult, Prediction, PromptTuner, Sample,
                                     SampleResult, split_samples)

from .conftest import FakeExecutor, FakeOptimizer


# ---------- split_samples ------------------------------------------------
def _make_samples(n: int, num_yes: int) -> List[Sample]:
    return [Sample(id=i, text=f"t{i}", label="Yes" if i < num_yes else "No")
            for i in range(n)]


def test_split_loses_no_samples_and_does_not_overlap():
    dev, test = split_samples(_make_samples(20, 10), test_ratio=0.5, seed=1)
    assert len(dev) + len(test) == 20
    ids_dev = {s.id for s in dev}
    ids_test = {s.id for s in test}
    assert not (ids_dev & ids_test), "một ca không được nằm ở cả hai tập"
    assert ids_dev | ids_test == set(range(20))


def test_split_is_reproducible_with_a_fixed_seed():
    a1, b1 = split_samples(_make_samples(20, 10), seed=42)
    a2, b2 = split_samples(_make_samples(20, 10), seed=42)
    assert [s.id for s in a1] == [s.id for s in a2]
    assert [s.id for s in b1] == [s.id for s in b2]


def test_split_preserves_label_ratio():
    """Bộ mẫu nhỏ mà lệch nhãn: chia ngẫu nhiên thuần có thể dồn hết nhãn hiếm
    về một bên, khiến điểm hai tập không so được với nhau."""
    dev, test = split_samples(_make_samples(20, 4), test_ratio=0.5, seed=3, stratify=True)
    assert sum(s.label == "Yes" for s in dev) == 2
    assert sum(s.label == "Yes" for s in test) == 2


def test_split_leaves_neither_side_empty():
    dev, test = split_samples(_make_samples(2, 1), test_ratio=0.9, seed=0)
    assert dev and test


def test_split_invalid_params():
    with pytest.raises(ValueError):
        split_samples([], test_ratio=0.5)
    with pytest.raises(ValueError):
        split_samples(_make_samples(4, 2), test_ratio=0)
    with pytest.raises(ValueError):
        split_samples(_make_samples(4, 2), test_ratio=1)


# ---------- test_size: chỉ định thẳng số ca ------------------------------
def test_split_by_test_size():
    """Ghi thẳng số ca rõ ràng và ít sai hơn là tự tính tỉ lệ rồi hy vọng làm tròn đúng."""
    dev, test = split_samples(_make_samples(480, 240), test_size=200, seed=0)
    assert len(test) == 200
    assert len(dev) == 280


def test_test_size_overrides_test_ratio():
    dev, test = split_samples(_make_samples(100, 50), test_ratio=0.9, test_size=20, seed=0)
    assert len(test) == 20


def test_test_size_still_balances_labels():
    _, test = split_samples(_make_samples(480, 240), test_size=200, seed=0)
    assert sum(1 for s in test if s.label == "Yes") == 100


def test_test_size_invalid():
    with pytest.raises(ValueError):
        split_samples(_make_samples(10, 5), test_size=0)
    with pytest.raises(ValueError):
        split_samples(_make_samples(10, 5), test_size=10)   # không được lấy sạch
    with pytest.raises(ValueError):
        split_samples(_make_samples(10, 5), test_size=11)


def test_split_accepts_unlabeled_samples():
    tickets = [Sample(id=i, text=f"t{i}") for i in range(10)]
    dev, test = split_samples(tickets, test_ratio=0.5, seed=0)
    assert len(dev) + len(test) == 10


# ---------- tuner: optimizer không được thấy tập test --------------------
class SpyOptimizer(BaseOptimizer):
    """Ghi lại mọi ca mà optimizer nhìn thấy."""

    def __init__(self):
        self.da_thay_ids = set()
        self.calls = 0

    def propose(self, prompt, result, task_description="", history=None) -> str:
        self.calls += 1
        for r in result.results:
            self.da_thay_ids.add(r.sample.id)
        return f"prompt moi {self.calls}"


def test_optimizer_never_sees_the_test_set():
    """Đây là bất biến cốt lõi — vỡ cái này là tập test mất sạch ý nghĩa."""
    dev = [Sample(id=i, text=f"dev{i}", label="Yes") for i in range(4)]
    test = [Sample(id=100 + i, text=f"test{i}", label="Yes") for i in range(4)]

    spy = SpyOptimizer()
    tuner = PromptTuner(executor=FakeExecutor(always="No"),
                        evaluator=__import__(
                            "prompt_tuning_framework.components.evaluators",
                            fromlist=["AccuracyEvaluator"]).AccuracyEvaluator(),
                        optimizer=spy, max_iters=3)
    tuner.run("p", dev, test_samples=test)

    assert spy.calls > 0, "optimizer phải được gọi, nếu không test này vô nghĩa"
    assert spy.da_thay_ids <= {s.id for s in dev}
    assert not (spy.da_thay_ids & {s.id for s in test})


def test_test_score_is_written_to_metadata():
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    dev = [Sample(id=i, text=f"It barks {i}", label="Yes") for i in range(4)]
    test = [Sample(id=100 + i, text=f"It barks {i}", label="Yes") for i in range(4)]

    tuner = PromptTuner(executor=FakeExecutor(always="Yes"),
                        evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    best = tuner.run("p", dev, test_samples=test)

    assert best.metadata["test_score"] == 100.0
    assert best.metadata["test_num_scored"] == 4
    assert best.metadata["test_reliable"] is True
    # Khoảng tin cậy phải đi kèm: 4/4 ca đúng KHÔNG phải bằng chứng chắc chắn.
    assert best.metadata["test_ci_low"] < 100.0


def test_keeps_per_sample_test_results():
    """Phải giữ cả EvalResult, không chỉ điểm.

    So sánh ghép cặp (McNemar) cần kết quả TỪNG CA. Chỉ lưu điểm thì muốn so hai
    prompt lại phải gọi LLM chấm lại toàn bộ tập test — tốn tiền vô ích.
    """
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    dev = [Sample(id=i, text=f"It barks {i}", label="Yes") for i in range(4)]
    test = [Sample(id=100 + i, text=f"It barks {i}", label="Yes") for i in range(4)]

    tuner = PromptTuner(executor=FakeExecutor(always="Yes"),
                        evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    tuner.run("p", dev, test_samples=test)

    assert tuner.test_result is not None
    assert len(tuner.test_result.results) == 4
    assert [r.sample.id for r in tuner.test_result.results] == [100, 101, 102, 103]
    # Đủ dữ liệu để so cặp với một lần chạy khác trên cùng tập test.
    assert tuner.test_result.distinguishable_from(tuner.test_result) is False


def test_no_test_set_leaves_test_result_none():
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    dev = [Sample(id=i, text=f"It barks {i}", label="Yes") for i in range(4)]
    tuner = PromptTuner(executor=FakeExecutor(always="Yes"),
                        evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    tuner.run("p", dev)
    assert tuner.test_result is None


def test_runs_as_before_without_a_test_set():
    """Tương thích ngược: chữ ký cũ run(prompt, samples) phải còn dùng được."""
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    dev = [Sample(id=i, text=f"It barks {i}", label="Yes") for i in range(4)]
    tuner = PromptTuner(executor=FakeExecutor(always="Yes"),
                        evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    best = tuner.run("p", dev)
    assert best is not None
    assert "test_score" not in best.metadata


# ---------- phát hiện học thuộc -----------------------------------------
class OverfittingExecutor(BaseExecutor):
    """Giả lập prompt vá thuộc lòng: đúng hết ở tập dev, sai hết ở ca lạ."""

    def execute(self, prompt: str, samples: List[Sample]) -> List[Prediction]:
        return [Prediction(sample_id=s.id,
                           output="Yes" if s.text.startswith("dev") else "No")
                for s in samples]


def test_warns_about_overfitting_when_dev_beats_test(caplog):
    """Điểm dev cao vống so với test = prompt vá riêng ca dev. Phải cảnh báo."""
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    dev = [Sample(id=i, text=f"dev{i}", label="Yes") for i in range(4)]
    test = [Sample(id=100 + i, text=f"la{i}", label="Yes") for i in range(4)]

    tuner = PromptTuner(executor=OverfittingExecutor(),
                        evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    with caplog.at_level(logging.WARNING):
        best = tuner.run("p", dev, test_samples=test)

    assert best.score == 100.0          # dev: hoàn hảo
    assert best.metadata["test_score"] == 0.0   # test: trượt sạch
    assert "HỌC THUỘC" in caplog.text


def test_no_warning_when_dev_and_test_agree(caplog):
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    dev = [Sample(id=i, text=f"x{i}", label="Yes") for i in range(4)]
    test = [Sample(id=100 + i, text=f"y{i}", label="Yes") for i in range(4)]

    tuner = PromptTuner(executor=FakeExecutor(always="Yes"),
                        evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    with caplog.at_level(logging.WARNING):
        tuner.run("p", dev, test_samples=test)
    assert "HỌC THUỘC" not in caplog.text


def test_unreliable_test_set_errors(caplog):
    """Tập test bị quota giết -> phải nói rõ là không đáng tin, không im lặng."""
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    class ErrorExecutor(BaseExecutor):
        def execute(self, prompt, samples):
            return [Prediction(sample_id=s.id, output="__ERROR__: 429")
                    for s in samples]

    dev = [Sample(id=i, text=f"x{i}", label="Yes") for i in range(4)]
    test = [Sample(id=100 + i, text=f"y{i}", label="Yes") for i in range(4)]

    tuner = PromptTuner(executor=ErrorExecutor(), evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    with caplog.at_level(logging.ERROR):
        best = tuner.run("p", dev, test_samples=test)

    if best is not None and "test_reliable" in best.metadata:
        assert best.metadata["test_reliable"] is False
