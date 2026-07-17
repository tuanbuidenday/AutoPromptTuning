"""Ví dụ 1 — Dùng nhanh framework: tối ưu prompt phân loại chó/mèo.

Chạy:  ./venv/bin/python -m prompt_tuning_framework.examples.quickstart
"""
import logging

from prompt_tuning_framework import PromptTuner, Sample
from prompt_tuning_framework.components import (AccuracyEvaluator, LLMExecutor,
                                                LLMRewriteOptimizer)

logging.basicConfig(level=logging.INFO, format="%(message)s")

LABELS = ["Yes", "No"]

# Bộ ca test: Yes = chó, No = mèo. Có vài ca "bẫy" cố ý.
SAMPLES = [
    Sample(id=0, text="It barks loudly whenever a stranger walks past the gate.", label="Yes"),
    Sample(id=1, text="It purrs on my lap and kneads the blanket with its paws.", label="No"),
    Sample(id=2, text="Loyal companion that fetches the ball every morning.", label="Yes"),
    Sample(id=3, text="It climbs the bookshelf and refuses to come down.", label="No"),
    Sample(id=4, text="We take it for a walk on a leash twice a day.", label="Yes"),
    Sample(id=5, text="It scratches the post and grooms itself for hours.", label="No"),
    Sample(id=6, text="A furry friend that wags its tail when I come home.", label="Yes"),
    Sample(id=7, text="Independent pet that meows at 5am demanding food.", label="No"),
    # Ca bẫy: mô tả chung chung, dễ đoán nhầm
    Sample(id=8, text="This animal is very affectionate and loves to nap in sunlight.", label="No"),
    Sample(id=9, text="It is a working animal often trained to guard the house.", label="Yes"),
]

TASK = ("Classify whether a short text describes a dog or a cat. "
        "Answer Yes if it describes a dog, No if it describes a cat.")

BAD_PROMPT = "Is this a dog? Yes or No"  # cố tình mơ hồ


def main():
    tuner = PromptTuner(
        executor=LLMExecutor(labels=LABELS),          # model rẻ mặc định của provider
        evaluator=AccuracyEvaluator(),
        optimizer=LLMRewriteOptimizer(labels=LABELS),  # model mạnh hơn để sửa prompt
        task_description=TASK,
        max_iters=3,
        target_score=100.0,
    )

    best = tuner.run(BAD_PROMPT, SAMPLES)

    print("\n" + "=" * 60)
    print("LỊCH SỬ CÁC PHIÊN BẢN PROMPT")
    print("=" * 60)
    for v in tuner.store.history():
        mark = " <-- TỐT NHẤT" if best and v.version == best.version else ""
        print(f"\n[v{v.version}] điểm: {v.score}/100{mark}\n{v.text}")

    if best:
        print("\n" + "=" * 60)
        print(f"KẾT QUẢ: {BAD_PROMPT!r} ({tuner.store.history()[0].score}/100)")
        print(f"     ->  {best.text!r} ({best.score}/100)")


if __name__ == "__main__":
    main()
