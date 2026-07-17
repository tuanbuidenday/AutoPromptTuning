"""Ví dụ KHÓ — phân loại ticket hỗ trợ: có khẩn cấp không?

Vì sao khó: "khẩn cấp" ở đây là QUY ĐỊNH NỘI BỘ của doanh nghiệp, không phải
kiến thức phổ thông. LLM không thể đoán ra, và các ca bẫy cố tình gào "URGENT!!!",
"CRITICAL" để dụ nó trả lời sai. Prompt ban đầu chắc chắn hỏng; framework phải
suy ra được quy định chỉ từ các CA SAI.

Quy định thật (KHÔNG hề nói cho model biết):
    Yes = khách ĐANG TRẢ TIỀN bị chặn hoàn toàn, ngay lúc này.
    No  = mọi thứ khác: lỗi giao diện, đòi tính năng, câu hỏi, có cách né,
          user free — dù ticket có gào to đến đâu.

Bộ mẫu (examples/tickets.csv) cân bằng 48 Yes / 48 No. Cân bằng là bắt buộc: nếu
để lệch 25/75 thì một prompt ngu ngốc luôn trả "No" đã được 75 điểm, và benchmark
sẽ không đo được gì.

Ví dụ này TÁCH dev/test. Optimizer chỉ nhìn thấy tập dev; điểm công bố lấy từ tập
test mà nó chưa từng thấy. Không tách thì 100/100 chỉ là điểm học thuộc: optimizer
được xem đúng các ca sai rồi viết prompt vá chúng.

Chi phí: mỗi vòng gọi LLM một lần cho MỖI ca dev (48 ca), cộng một lần chấm tập
test ở cuối. Với free tier có giới hạn theo ngày, chạy đủ 4 vòng gần như chắc chắn
sẽ cạn quota — hãy giảm max_iters hoặc dùng bộ nhỏ hơn nếu chỉ muốn thử.

Chạy:  ./venv/bin/python -m prompt_tuning_framework.examples.hard_example
"""
import logging
from pathlib import Path

from prompt_tuning_framework import (PromptTuner, Sample, load_samples_csv,
                                     split_samples)
from prompt_tuning_framework.components import (AccuracyEvaluator, LLMExecutor,
                                                LLMRewriteOptimizer)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

LABELS = ["Yes", "No"]
DATASET = Path(__file__).parent / "tickets.csv"

TASK = ("Classify whether a customer support ticket must be escalated as urgent. "
        "Answer Yes if it is urgent, No if it is not.")

# Prompt mơ hồ: không hề nêu quy định -> sẽ bị các ca gào 'URGENT' đánh lừa
BAD_PROMPT = "Is this support ticket urgent? Yes or No"


def main(max_iters: int = 4):
    samples = load_samples_csv(str(DATASET))
    # seed cố định -> chia lại y hệt, để so sánh giữa các lần chạy là công bằng.
    dev, test = split_samples(samples, test_ratio=0.5, seed=0)
    print(f"Bộ mẫu: {len(samples)} ca  ->  dev {len(dev)} / test {len(test)} "
          f"(test được giữ riêng, optimizer không thấy)")

    tuner = PromptTuner(
        # delay=5.0: free tier Gemini chỉ cho 15 request/phút. Không tiết chế thì
        # ca lỗi 429 sẽ bị loại khỏi mẫu số -> điểm bị thổi phồng thành 100 giả.
        executor=LLMExecutor(labels=LABELS, delay=5.0),  # model rẻ — vai "thí sinh"
        evaluator=AccuracyEvaluator(),
        optimizer=LLMRewriteOptimizer(labels=LABELS),    # model mạnh — vai "người sửa prompt"
        task_description=TASK,
        max_iters=max_iters,
        target_score=100.0,
    )

    best = tuner.run(BAD_PROMPT, dev, test_samples=test)
    history = tuner.store.history()
    if best is None:
        print("\nKhông có kết quả đáng tin (xem log lỗi ở trên).")
        return

    print("\n" + "=" * 70)
    print("LỊCH SỬ CÁC PHIÊN BẢN PROMPT (điểm trên tập DEV)")
    print("=" * 70)
    for v in history:
        mark = "  <-- TỐT NHẤT" if v.version == best.version else ""
        print(f"\n[v{v.version}] {v.score}/100{mark}\n{v.text}")

    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)
    dau = history[0].score
    print(f"Tập DEV  (optimizer đã thấy) : {dau}/100  ->  {best.score}/100")

    md = best.metadata
    if "test_score" in md:
        lo, hi = md["test_ci_low"], md["test_ci_high"]
        canh = "" if md.get("test_reliable") else "   <-- KHÔNG ĐÁNG TIN"
        print(f"Tập TEST (chưa từng thấy)    : {md['test_score']}/100{canh}")
        print(f"  khoảng tin cậy 95%         : [{lo}, {hi}] trên {md['test_num_scored']} ca")
        print("\nCon số đáng công bố là điểm TEST, không phải điểm DEV.")
        print("Và phải công bố kèm khoảng tin cậy: điểm trần tạo cảm giác chắc chắn")
        print("không có thật — trên bộ mẫu cỡ này, sai số vẫn còn khá rộng.")
    print("=" * 70)


if __name__ == "__main__":
    main()
