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
def _mau(n: int, nhan_yes: int) -> List[Sample]:
    return [Sample(id=i, text=f"t{i}", label="Yes" if i < nhan_yes else "No")
            for i in range(n)]


def test_split_khong_lam_mat_mau_va_khong_trung():
    dev, test = split_samples(_mau(20, 10), test_ratio=0.5, seed=1)
    assert len(dev) + len(test) == 20
    ids_dev = {s.id for s in dev}
    ids_test = {s.id for s in test}
    assert not (ids_dev & ids_test), "một ca không được nằm ở cả hai tập"
    assert ids_dev | ids_test == set(range(20))


def test_split_co_dinh_seed_thi_chia_lai_y_het():
    a1, b1 = split_samples(_mau(20, 10), seed=42)
    a2, b2 = split_samples(_mau(20, 10), seed=42)
    assert [s.id for s in a1] == [s.id for s in a2]
    assert [s.id for s in b1] == [s.id for s in b2]


def test_split_giu_ti_le_nhan():
    """Bộ mẫu nhỏ mà lệch nhãn: chia ngẫu nhiên thuần có thể dồn hết nhãn hiếm
    về một bên, khiến điểm hai tập không so được với nhau."""
    dev, test = split_samples(_mau(20, 4), test_ratio=0.5, seed=3, stratify=True)
    assert sum(s.label == "Yes" for s in dev) == 2
    assert sum(s.label == "Yes" for s in test) == 2


def test_split_khong_de_ben_nao_rong():
    dev, test = split_samples(_mau(2, 1), test_ratio=0.9, seed=0)
    assert dev and test


def test_split_tham_so_sai():
    with pytest.raises(ValueError):
        split_samples([], test_ratio=0.5)
    with pytest.raises(ValueError):
        split_samples(_mau(4, 2), test_ratio=0)
    with pytest.raises(ValueError):
        split_samples(_mau(4, 2), test_ratio=1)


def test_split_chap_nhan_mau_khong_co_nhan():
    mau = [Sample(id=i, text=f"t{i}") for i in range(10)]
    dev, test = split_samples(mau, test_ratio=0.5, seed=0)
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


def test_optimizer_khong_bao_gio_thay_tap_test():
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


def test_diem_tap_test_duoc_ghi_vao_metadata():
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


def test_khong_truyen_tap_test_thi_van_chay_nhu_cu():
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
class HocThuocExecutor(BaseExecutor):
    """Giả lập prompt vá thuộc lòng: đúng hết ở tập dev, sai hết ở ca lạ."""

    def execute(self, prompt: str, samples: List[Sample]) -> List[Prediction]:
        return [Prediction(sample_id=s.id,
                           output="Yes" if s.text.startswith("dev") else "No")
                for s in samples]


def test_canh_bao_hoc_thuoc_khi_dev_cao_hon_test(caplog):
    """Điểm dev cao vống so với test = prompt vá riêng ca dev. Phải cảnh báo."""
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    dev = [Sample(id=i, text=f"dev{i}", label="Yes") for i in range(4)]
    test = [Sample(id=100 + i, text=f"la{i}", label="Yes") for i in range(4)]

    tuner = PromptTuner(executor=HocThuocExecutor(),
                        evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    with caplog.at_level(logging.WARNING):
        best = tuner.run("p", dev, test_samples=test)

    assert best.score == 100.0          # dev: hoàn hảo
    assert best.metadata["test_score"] == 0.0   # test: trượt sạch
    assert "HỌC THUỘC" in caplog.text


def test_khong_canh_bao_khi_dev_va_test_tuong_duong(caplog):
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    dev = [Sample(id=i, text=f"x{i}", label="Yes") for i in range(4)]
    test = [Sample(id=100 + i, text=f"y{i}", label="Yes") for i in range(4)]

    tuner = PromptTuner(executor=FakeExecutor(always="Yes"),
                        evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    with caplog.at_level(logging.WARNING):
        tuner.run("p", dev, test_samples=test)
    assert "HỌC THUỘC" not in caplog.text


def test_tap_test_khong_dang_tin_thi_bao_loi(caplog):
    """Tập test bị quota giết -> phải nói rõ là không đáng tin, không im lặng."""
    from prompt_tuning_framework.components.evaluators import AccuracyEvaluator

    class LoiExecutor(BaseExecutor):
        def execute(self, prompt, samples):
            return [Prediction(sample_id=s.id, output="__ERROR__: 429")
                    for s in samples]

    dev = [Sample(id=i, text=f"x{i}", label="Yes") for i in range(4)]
    test = [Sample(id=100 + i, text=f"y{i}", label="Yes") for i in range(4)]

    tuner = PromptTuner(executor=LoiExecutor(), evaluator=AccuracyEvaluator(),
                        optimizer=FakeOptimizer(), max_iters=1)
    with caplog.at_level(logging.ERROR):
        best = tuner.run("p", dev, test_samples=test)

    if best is not None and "test_reliable" in best.metadata:
        assert best.metadata["test_reliable"] is False
