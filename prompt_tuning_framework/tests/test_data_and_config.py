"""Test nạp dataset và dựng tuner từ YAML (config-driven)."""
import pytest

from prompt_tuning_framework import tuner_from_config
from prompt_tuning_framework.components import (AccuracyEvaluator,
                                                InMemoryPromptStore,
                                                SQLitePromptStore)
from prompt_tuning_framework.data import load_samples_csv

CSV = "text,label\nIt barks.,Yes\nIt purrs.,No\n"


def _write(tmp_path, content, name="d.csv"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------- data loader ----------
def test_load_csv(tmp_path):
    s = load_samples_csv(_write(tmp_path, CSV))
    assert [x.text for x in s] == ["It barks.", "It purrs."]
    assert [x.label for x in s] == ["Yes", "No"]
    assert [x.id for x in s] == [0, 1]


def test_load_csv_chap_nhan_cot_annotation(tmp_path):
    """Định dạng dump của AutoPrompt dùng cột 'annotation'."""
    s = load_samples_csv(_write(tmp_path, "text,annotation\nabc,Yes\n"))
    assert s[0].label == "Yes"


def test_load_csv_bo_qua_dong_rong(tmp_path):
    s = load_samples_csv(_write(tmp_path, "text,label\nabc,Yes\n,No\n"))
    assert len(s) == 1


def test_load_csv_thieu_cot_text_bao_loi(tmp_path):
    with pytest.raises(ValueError, match="Thiếu cột"):
        load_samples_csv(_write(tmp_path, "noidung,label\nabc,Yes\n"))


def test_load_csv_khong_thay_file():
    with pytest.raises(FileNotFoundError):
        load_samples_csv("/khong/ton/tai.csv")


# ---------- config-driven ----------
def test_tuner_from_config_dung_dung_component(tmp_path):
    cfg = {
        "task_description": "phan loai",
        "max_iters": 7,
        "target_score": 95,
        "store": {"name": "sqlite",
                  "params": {"db_path": str(tmp_path / "c.db"), "run_name": "r"}},
        "executor": {"name": "fake_cfg"},
        "evaluator": {"name": "accuracy"},
        "optimizer": {"name": "fake_cfg"},
    }
    # đăng ký component giả để không phải gọi LLM
    from prompt_tuning_framework import BaseExecutor, BaseOptimizer, register

    @register("executor", "fake_cfg")
    class E(BaseExecutor):
        def execute(self, prompt, samples):
            return []

    @register("optimizer", "fake_cfg")
    class O(BaseOptimizer):
        def propose(self, prompt, result, task_description="", history=None):
            return ""

    t = tuner_from_config(cfg)
    assert isinstance(t.store, SQLitePromptStore)
    assert isinstance(t.evaluator, AccuracyEvaluator)
    assert t.max_iters == 7 and t.target_score == 95
    assert t.task_description == "phan loai"


def test_config_thieu_name_bao_loi():
    with pytest.raises(ValueError, match="Thiếu 'name'"):
        tuner_from_config({"executor": {"params": {}}})


def test_config_khong_khai_store_thi_dung_memory():
    from prompt_tuning_framework import BaseExecutor, BaseOptimizer, register

    @register("executor", "e2")
    class E(BaseExecutor):
        def execute(self, prompt, samples):
            return []

    @register("optimizer", "o2")
    class O(BaseOptimizer):
        def propose(self, prompt, result, task_description="", history=None):
            return ""

    t = tuner_from_config({"executor": {"name": "e2"}, "evaluator": {"name": "accuracy"},
                           "optimizer": {"name": "o2"}})
    assert isinstance(t.store, InMemoryPromptStore)
