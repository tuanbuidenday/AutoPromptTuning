"""Test cơ chế plugin registry."""
import pytest

from prompt_tuning_framework import BaseEvaluator, available, create, get, register
from prompt_tuning_framework.core import registry


@pytest.fixture(autouse=True)
def _restore_registry():
    """Giữ registry sạch giữa các test."""
    snapshot = {k: dict(v) for k, v in registry._REGISTRY.items()}
    yield
    for k, v in snapshot.items():
        registry._REGISTRY[k] = v


class Dummy(BaseEvaluator):
    def __init__(self, factor=1):
        self.factor = factor

    def evaluate(self, prompt, predictions, samples):
        raise NotImplementedError


def test_builtin_plugins_are_registered():
    assert "memory" in available("store")
    assert "sqlite" in available("store")
    assert "llm" in available("executor")
    assert "accuracy" in available("evaluator")
    assert "llm_rewrite" in available("optimizer")


def test_register_and_create():
    register("evaluator", "dummy", Dummy)
    assert "dummy" in available("evaluator")
    obj = create("evaluator", "dummy", factor=7)
    assert isinstance(obj, Dummy) and obj.factor == 7


def test_register_as_decorator():
    @register("evaluator", "dummy_dec")
    class DummyDec(Dummy):
        pass

    assert get("evaluator", "dummy_dec") is DummyDec


def test_create_unknown_name_raises():
    with pytest.raises(KeyError, match="Chưa đăng ký"):
        create("evaluator", "khong-ton-tai")


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind phải thuộc"):
        register("khong-hop-le", "x", Dummy)
