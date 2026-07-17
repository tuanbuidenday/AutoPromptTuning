"""Ví dụ 2 — Cắm component TỰ VIẾT vào framework (không cần gọi LLM).

Chứng minh 2 điều:
  1. Inversion of Control: framework gọi ngược lại code của bạn.
  2. Mọi bước đều thay thế được — ở đây ta thay Executor và Optimizer bằng
     phiên bản giả lập, chạy offline, không tốn tiền API.

Chạy:  ./venv/bin/python -m prompt_tuning_framework.examples.custom_components
"""
import logging
from typing import List, Optional

from prompt_tuning_framework import (BaseCallback, BaseExecutor, BaseOptimizer,
                                     EvalResult, Prediction, PromptTuner,
                                     PromptVersion, Sample, available, register)
from prompt_tuning_framework.components import AccuracyEvaluator, InMemoryPromptStore

logging.basicConfig(level=logging.INFO, format="%(message)s")

LABELS = ["Yes", "No"]

SAMPLES = [
    Sample(id=0, text="It barks at strangers.", label="Yes"),
    Sample(id=1, text="It purrs on my lap.", label="No"),
    Sample(id=2, text="We walk it on a leash.", label="Yes"),
    Sample(id=3, text="It climbs the bookshelf.", label="No"),
]

# Từ khoá mà một prompt "tốt" cần nêu ra để phân biệt chó/mèo
DOG_WORDS = {"bark", "leash", "walk", "fetch", "wag"}


@register("executor", "fake")
class FakeExecutor(BaseExecutor):
    """Giả lập LLM: prompt càng nêu rõ đặc điểm chó thì đoán càng đúng.

    Prompt mơ hồ -> đoán bừa 'Yes'. Prompt có nêu từ khoá -> suy luận đúng.
    """

    def execute(self, prompt: str, samples: List[Sample]) -> List[Prediction]:
        informed = any(w in prompt.lower() for w in DOG_WORDS)
        out = []
        for s in samples:
            if not informed:
                guess = "Yes"  # prompt dở: đoán bừa
            else:
                guess = "Yes" if any(w in s.text.lower() for w in DOG_WORDS) else "No"
            out.append(Prediction(sample_id=s.id, output=guess))
        return out


@register("optimizer", "fake")
class FakeOptimizer(BaseOptimizer):
    """Giả lập optimizer: nhìn lỗi rồi bổ sung đặc điểm nhận dạng vào prompt."""

    def propose(self, prompt: str, result: EvalResult, task_description: str = "",
                history: Optional[List[PromptVersion]] = None) -> str:
        if not result.errors:
            return ""
        return ("Decide if the text describes a dog or a cat. "
                "A dog barks, walks on a leash, fetches, and wags its tail. "
                "A cat purrs, climbs, and grooms itself. Answer Yes for dog, No for cat.")


class PrintProgress(BaseCallback):
    """Hook theo dõi — framework gọi lại sau mỗi vòng."""

    def on_iteration_end(self, iteration: int, version: PromptVersion,
                         result: EvalResult) -> None:
        print(f"  [hook] vòng {iteration}: {result.score}/100, "
              f"{len(result.errors)} ca sai")


def main():
    print("Plugin đã đăng ký:")
    for kind in ("store", "executor", "evaluator", "optimizer"):
        print(f"  {kind:10}: {available(kind)}")

    tuner = PromptTuner(
        executor=FakeExecutor(),                 # component TỰ VIẾT
        evaluator=AccuracyEvaluator(),           # component sẵn có
        optimizer=FakeOptimizer(),               # component TỰ VIẾT
        store=InMemoryPromptStore(),
        task_description="Classify dog vs cat.",
        max_iters=3,
        callbacks=[PrintProgress()],
    )

    print("\nChạy vòng lặp tối ưu...")
    best = tuner.run("Is this a dog? Yes or No", SAMPLES)

    print("\n=== Các phiên bản prompt ===")
    for v in tuner.store.history():
        star = " <-- TỐT NHẤT" if best and v.version == best.version else ""
        print(f"[v{v.version}] {v.score}/100{star}: {v.text[:70]}")

    first = tuner.store.history()[0]
    assert best is not None and best.score == 100.0, "Framework phải tối ưu lên 100"
    print(f"\n✅ Framework hoạt động: prompt dở {first.score}/100 "
          f"-> prompt tốt {best.score}/100")


if __name__ == "__main__":
    main()
