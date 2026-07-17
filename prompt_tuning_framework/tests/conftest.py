"""Component giả lập dùng chung cho test — không gọi LLM, chạy offline."""
from typing import List, Optional

import pytest

from prompt_tuning_framework import (BaseEvaluator, BaseExecutor, BaseOptimizer,
                                     EvalResult, Prediction, PromptVersion,
                                     Sample, SampleResult)

LABELS = ["Yes", "No"]
DOG_WORDS = {"bark", "leash"}


@pytest.fixture
def samples() -> List[Sample]:
    return [
        Sample(id=0, text="It barks at strangers.", label="Yes"),
        Sample(id=1, text="It purrs on my lap.", label="No"),
        Sample(id=2, text="We walk it on a leash.", label="Yes"),
        Sample(id=3, text="It climbs the shelf.", label="No"),
    ]


class FakeExecutor(BaseExecutor):
    """Prompt có nêu từ khoá -> đoán đúng; prompt mơ hồ -> luôn đoán 'Yes'."""

    def __init__(self, always: Optional[str] = None):
        self.always = always
        self.calls = 0

    def execute(self, prompt: str, samples: List[Sample]) -> List[Prediction]:
        self.calls += 1
        informed = any(w in prompt.lower() for w in DOG_WORDS)
        out = []
        for s in samples:
            if self.always is not None:
                guess = self.always
            elif not informed:
                guess = "Yes"
            else:
                guess = "Yes" if any(w in s.text.lower() for w in DOG_WORDS) else "No"
            out.append(Prediction(sample_id=s.id, output=guess))
        return out


class FakeOptimizer(BaseOptimizer):
    """Đề xuất prompt có từ khoá; có thể giả lập 'bó tay' (trả rỗng)."""

    def __init__(self, give_up: bool = False):
        self.give_up = give_up
        self.calls = 0

    def propose(self, prompt, result, task_description="", history=None) -> str:
        self.calls += 1
        if self.give_up:
            return ""
        # Mỗi lần đề xuất một prompt KHÁC nhau (optimizer thật cũng vậy) —
        # nếu trả trùng prompt cũ, tuner sẽ dừng vì coi như bó tay.
        return ("A dog barks or walks on a leash; a cat does not. "
                f"Answer Yes or No. (rev {self.calls})")


class ConstantEvaluator(BaseEvaluator):
    """Luôn trả về một điểm cố định — để test tiêu chí dừng."""

    def __init__(self, score: float):
        self.score = score

    def evaluate(self, prompt, predictions, samples) -> EvalResult:
        return EvalResult(
            score=self.score,
            results=[SampleResult(sample=s, predicted="X", expected=s.label,
                                  correct=False) for s in samples],
        )


@pytest.fixture
def fake_executor() -> FakeExecutor:
    return FakeExecutor()


@pytest.fixture
def fake_optimizer() -> FakeOptimizer:
    return FakeOptimizer()
