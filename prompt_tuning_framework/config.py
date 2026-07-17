"""Dựng PromptTuner từ file YAML — người dùng khai báo component theo tên.

Ví dụ config:

    task_description: "Classify whether a text describes a dog."
    max_iters: 5
    target_score: 100
    store:
        name: sqlite
        params: {db_path: runs.db, run_name: dog-vs-cat}
    executor:
        name: llm
        params: {model: gemini-2.5-flash-lite, labels: ["Yes", "No"]}
    evaluator:
        name: accuracy
    optimizer:
        name: autoprompt          # đổi thành llm_rewrite là xong, không sửa code
        params: {model: gemini-2.5-flash, labels: ["Yes", "No"]}
"""
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from . import components  # noqa: F401  (import để component tự đăng ký)
from .core.registry import create
from .core.tuner import PromptTuner


def _build(kind: str, spec: Optional[Dict[str, Any]]):
    if not spec:
        return None
    name = spec.get("name")
    if not name:
        raise ValueError(f"Thiếu 'name' cho {kind} trong config.")
    return create(kind, name, **(spec.get("params") or {}))


def tuner_from_config(config: Dict[str, Any]) -> PromptTuner:
    """Dựng PromptTuner từ dict config."""
    return PromptTuner(
        executor=_build("executor", config.get("executor")),
        evaluator=_build("evaluator", config.get("evaluator")),
        optimizer=_build("optimizer", config.get("optimizer")),
        store=_build("store", config.get("store")),
        task_description=config.get("task_description", ""),
        max_iters=config.get("max_iters", 5),
        target_score=config.get("target_score", 100.0),
        patience=config.get("patience"),
        min_delta=config.get("min_delta", 0.01),
    )


def tuner_from_yaml(path: str) -> PromptTuner:
    """Dựng PromptTuner từ file YAML."""
    with open(Path(path)) as f:
        return tuner_from_config(yaml.safe_load(f) or {})
