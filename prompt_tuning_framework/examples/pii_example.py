"""Ví dụ 2: phát hiện văn bản LỘ THÔNG TIN KHÁCH HÀNG (tiếng Việt).

Cho thấy framework không gắn với một bài toán nào: đổi bộ mẫu + task_description
là xong, không sửa dòng nào trong lõi. Bài này khác `hard_example.py` ở ba chỗ —
tiếng Việt, miền dữ liệu khác, và quy định ẩn thuộc loại khác.

Prompt khởi đầu cố tình mơ hồ, đúng như người ta hay viết:
    "Dữ liệu đầu vào là nhạy cảm, lộ thông tin khách hàng, trả lời Yes or No"
Nó không hề định nghĩa "lộ" là gì, nên model sẽ bám vào chữ "nhạy cảm" — mà bộ
mẫu đã rải chữ đó đều 50/50 giữa Yes và No, nên bám vào là sai một nửa.

Chạy:
    python -m prompt_tuning_framework.examples.pii_example
    python -m prompt_tuning_framework.examples.pii_example --nho 24 --max-iters 2
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

from prompt_tuning_framework import (PromptTuner, load_samples_csv,
                                     split_samples)
from prompt_tuning_framework.components import (AccuracyEvaluator, LLMExecutor,
                                                LLMRewriteOptimizer)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

LABELS = ["Yes", "No"]
DATASET = Path(__file__).parent / "pii.csv"

# 120 ca chia đôi. Nhỏ hơn bộ ticket (480) nên khoảng tin cậy rộng hơn — 60 ca
# đạt 100 điểm chỉ chứng minh được >= ~94%, chứ không phải >= 98% như bộ 200 ca.
N_TEST = 60

TASK = ("Classify whether a Vietnamese text leaks a customer's personal "
        "identifying data. Answer Yes if it does, No if it does not.")

BAD_PROMPT = "Dữ liệu đầu vào là nhạy cảm, lộ thông tin khách hàng, trả lời Yes or No"


def main(max_iters: int = 3, n_mau: Optional[int] = None,
         delay: float = 0.0, num_workers: int = 8):
    samples = load_samples_csv(str(DATASET))
    if n_mau:
        samples = samples[:n_mau]
    dev, test = split_samples(samples, test_size=min(N_TEST, len(samples) // 2),
                              seed=0)
    print(f"Nạp {len(samples)} ca — {len(dev)} train / {len(test)} test\n")

    tuner = PromptTuner(
        executor=LLMExecutor(labels=LABELS, delay=delay, num_workers=num_workers),
        evaluator=AccuracyEvaluator(),
        optimizer=LLMRewriteOptimizer(labels=LABELS),
        task_description=TASK,
        max_iters=max_iters,
        target_score=100.0,
    )

    best = tuner.run(BAD_PROMPT, dev, test_samples=test)
    if best is None:
        print("\nKhông có kết quả đáng tin (xem log lỗi ở trên).")
        return

    history = tuner.store.history()
    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)
    print(f"\nTRƯỚC ({history[0].score}/100 train):\n  {BAD_PROMPT}")
    print(f"\nSAU ({best.score}/100 train):\n  {best.text}")

    md = best.metadata
    if "test_score" in md:
        lo, hi = md["test_ci_low"], md["test_ci_high"]
        print(f"\nTẬP TEST (optimizer chưa từng thấy): {md['test_score']}/100")
        print(f"  khoảng tin cậy 95%: [{lo:.1f}, {hi:.1f}] trên {md['test_num_scored']} ca")
        print(f"  -> chứng minh được prompt đúng ít nhất {lo:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nho", type=int, default=None, metavar="N",
                   help="chỉ dùng N ca đầu (chạy thử cho rẻ)")
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    main(max_iters=a.max_iters, n_mau=a.nho, delay=a.delay, num_workers=a.workers)
