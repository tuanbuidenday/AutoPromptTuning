"""Prompt Tuning Framework — bộ khung tối ưu prompt tự động/bán tự động.

Vòng lặp khép kín 4 bước, framework giữ quyền điều khiển (Inversion of Control):
    ① Quản lý Prompt  -> BasePromptStore
    ② Thực thi        -> BaseExecutor
    ③ Đánh giá        -> BaseEvaluator
    ④ Tối ưu hóa      -> BaseOptimizer

Dùng nhanh:

    from prompt_tuning_framework import PromptTuner, Sample
    from prompt_tuning_framework.components import (
        LLMExecutor, AccuracyEvaluator, LLMRewriteOptimizer)

    tuner = PromptTuner(
        executor=LLMExecutor(labels=["Yes", "No"]),
        evaluator=AccuracyEvaluator(),
        optimizer=LLMRewriteOptimizer(labels=["Yes", "No"]),
        task_description="Classify whether the text describes a dog.",
    )
    best = tuner.run("Is this a dog? Yes or No", samples)

Mở rộng (cắm component của bạn):

    from prompt_tuning_framework import BaseEvaluator, register

    @register("evaluator", "my_eval")
    class MyEvaluator(BaseEvaluator):
        def evaluate(self, prompt, predictions, samples): ...
"""
from . import components  # noqa: F401  (nạp để component tự đăng ký)
from .config import tuner_from_config, tuner_from_yaml
from .core import (BaseCallback, BaseEvaluator, BaseExecutor, BaseOptimizer,
                   BasePromptStore, EvalResult, Prediction, PromptTuner,
                   PromptVersion, Sample, SampleResult, available, create,
                   get, register)

__version__ = "0.1.0"

__all__ = [
    # vòng lặp
    "PromptTuner",
    # 4 điểm mở rộng
    "BasePromptStore", "BaseExecutor", "BaseEvaluator", "BaseOptimizer", "BaseCallback",
    # kiểu dữ liệu
    "Sample", "Prediction", "SampleResult", "EvalResult", "PromptVersion",
    # plugin registry
    "register", "create", "available", "get",
    # config-driven
    "tuner_from_config", "tuner_from_yaml",
    "__version__",
]
