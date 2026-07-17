"""Test vòng lặp PromptTuner — trái tim IoC của framework."""
import pytest

from prompt_tuning_framework import BaseCallback, PromptTuner
from prompt_tuning_framework.components import AccuracyEvaluator, InMemoryPromptStore

from .conftest import ConstantEvaluator, FakeExecutor, FakeOptimizer

BAD_PROMPT = "Is this a dog? Yes or No"


def _tuner(executor=None, evaluator=None, optimizer=None, **kw):
    return PromptTuner(
        executor=executor or FakeExecutor(),
        evaluator=evaluator or AccuracyEvaluator(),
        optimizer=optimizer or FakeOptimizer(),
        store=kw.pop("store", InMemoryPromptStore()),
        task_description="Classify dog vs cat.",
        **kw,
    )


def test_toi_uu_cai_thien_diem(samples):
    """Prompt dở phải được sửa thành prompt tốt hơn."""
    t = _tuner(max_iters=3)
    best = t.run(BAD_PROMPT, samples)

    hist = t.store.history()
    assert hist[0].text == BAD_PROMPT
    assert hist[0].score == 50.0        # prompt dở: đoán bừa 'Yes'
    assert best.score == 100.0          # prompt mới: đúng hết
    assert best.score > hist[0].score


def test_framework_goi_nguoc_component_cua_nguoi_dung(samples):
    """Chứng minh IoC: tuner phải GỌI executor/optimizer của người dùng."""
    ex, opt = FakeExecutor(), FakeOptimizer()
    _tuner(executor=ex, optimizer=opt, max_iters=3).run(BAD_PROMPT, samples)
    assert ex.calls >= 2   # chạy prompt gốc + prompt mới
    assert opt.calls >= 1  # được nhờ đề xuất ít nhất 1 lần


def test_dung_som_khi_dat_target_score(samples):
    t = _tuner(max_iters=10)
    t.run(BAD_PROMPT, samples)
    # vòng 0 = 50, vòng 1 = 100 -> dừng ngay, chỉ có 2 phiên bản
    assert len(t.store.history()) == 2


def test_ton_trong_max_iters(samples):
    """Điểm không bao giờ đạt target -> phải dừng đúng số vòng."""
    t = _tuner(evaluator=ConstantEvaluator(10.0), max_iters=3)
    t.run(BAD_PROMPT, samples)
    assert len(t.store.history()) == 3


def test_dung_khi_optimizer_bo_cuoc(samples):
    """Optimizer trả rỗng -> dừng an toàn, không sập."""
    t = _tuner(optimizer=FakeOptimizer(give_up=True), max_iters=5)
    best = t.run(BAD_PROMPT, samples)
    assert len(t.store.history()) == 1
    assert best.text == BAD_PROMPT


def test_dung_khi_optimizer_lap_lai_prompt_cu(samples):
    """Optimizer đề xuất trùng prompt hiện tại -> dừng, không lặp vô ích."""
    from prompt_tuning_framework import BaseOptimizer

    class Repeats(BaseOptimizer):
        def propose(self, prompt, result, task_description="", history=None):
            return prompt  # trả y hệt prompt cũ

    t = _tuner(evaluator=ConstantEvaluator(10.0), optimizer=Repeats(), max_iters=5)
    t.run(BAD_PROMPT, samples)
    assert len(t.store.history()) == 1


def test_patience_dung_khi_khong_cai_thien(samples):
    t = _tuner(evaluator=ConstantEvaluator(10.0), max_iters=10, patience=2)
    t.run(BAD_PROMPT, samples)
    assert len(t.store.history()) < 10  # dừng sớm nhờ patience


def test_moi_phien_ban_deu_duoc_cham_diem(samples):
    t = _tuner(max_iters=3)
    t.run(BAD_PROMPT, samples)
    assert all(v.score is not None for v in t.store.history())


def test_tra_ve_phien_ban_tot_nhat_khong_phai_cuoi_cung(samples):
    """Điểm có thể tụt; best() phải trả bản điểm cao nhất."""
    class Dropping(ConstantEvaluator):
        def __init__(self):
            self.scores = iter([80.0, 20.0, 10.0])

        def evaluate(self, prompt, predictions, samples):
            r = ConstantEvaluator(next(self.scores)).evaluate(prompt, predictions, samples)
            return r

    t = _tuner(evaluator=Dropping(), max_iters=3)
    best = t.run(BAD_PROMPT, samples)
    assert best.score == 80.0 and best.version == 0


def test_callback_duoc_goi(samples):
    events = []

    class Spy(BaseCallback):
        def on_run_start(self, prompt, samples):
            events.append("start")

        def on_iteration_end(self, iteration, version, result):
            events.append(f"iter{iteration}")

        def on_run_end(self, best):
            events.append("end")

    _tuner(max_iters=2, callbacks=[Spy()]).run(BAD_PROMPT, samples)
    assert events[0] == "start" and events[-1] == "end"
    assert "iter0" in events


def test_callback_loi_khong_lam_sap_vong_lap(samples):
    class Broken(BaseCallback):
        def on_iteration_end(self, iteration, version, result):
            raise RuntimeError("callback hỏng")

    best = _tuner(max_iters=2, callbacks=[Broken()]).run(BAD_PROMPT, samples)
    assert best is not None  # vẫn chạy xong


def test_dataset_rong_bao_loi():
    with pytest.raises(ValueError, match="ít nhất 1 sample"):
        _tuner().run(BAD_PROMPT, [])


def test_store_mac_dinh_la_memory(samples):
    t = PromptTuner(executor=FakeExecutor(), evaluator=AccuracyEvaluator(),
                    optimizer=FakeOptimizer(), max_iters=1)
    t.run(BAD_PROMPT, samples)
    assert isinstance(t.store, InMemoryPromptStore)
