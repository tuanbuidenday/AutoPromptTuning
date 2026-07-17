"""Registry plugin: đăng ký component theo tên để dựng được từ config/YAML.

Nhờ đó người dùng khai báo trong YAML:
    optimizer:
        name: 'autoprompt'
        params: {...}
và framework tự dựng đúng lớp — không cần sửa code lõi.
"""
from typing import Any, Callable, Dict, List, Type

KINDS = ("store", "executor", "evaluator", "optimizer")

_REGISTRY: Dict[str, Dict[str, Type]] = {k: {} for k in KINDS}


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind phải thuộc {KINDS}, nhận được: {kind!r}")


def register(kind: str, name: str, cls: Type = None) -> Any:
    """Đăng ký một component. Dùng trực tiếp hoặc làm decorator.

        register("optimizer", "my_opt", MyOptimizer)

        @register("optimizer", "my_opt")
        class MyOptimizer(BaseOptimizer): ...
    """
    _check_kind(kind)

    def _do(target: Type) -> Type:
        _REGISTRY[kind][name] = target
        return target

    return _do if cls is None else _do(cls)


def create(kind: str, name: str, **params: Any) -> Any:
    """Dựng một component đã đăng ký theo tên."""
    _check_kind(kind)
    if name not in _REGISTRY[kind]:
        raise KeyError(
            f"Chưa đăng ký {kind} tên {name!r}. Hiện có: {available(kind)}"
        )
    return _REGISTRY[kind][name](**params)


def available(kind: str) -> List[str]:
    """Liệt kê tên các component đã đăng ký của một loại."""
    _check_kind(kind)
    return sorted(_REGISTRY[kind])


def get(kind: str, name: str) -> Type:
    """Lấy lớp (không khởi tạo)."""
    _check_kind(kind)
    return _REGISTRY[kind][name]
