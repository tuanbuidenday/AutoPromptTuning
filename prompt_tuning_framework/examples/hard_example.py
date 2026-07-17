"""Ví dụ KHÓ — phân loại ticket hỗ trợ: có khẩn cấp không?

Vì sao khó: "khẩn cấp" ở đây là QUY ĐỊNH NỘI BỘ của doanh nghiệp, không phải
kiến thức phổ thông. LLM không thể đoán ra, và các ca bẫy cố tình gào "URGENT!!!",
"CRITICAL" để dụ nó trả lời sai. Prompt ban đầu chắc chắn hỏng; framework phải
suy ra được quy định chỉ từ các CA SAI.

Quy định thật (KHÔNG hề nói cho model biết):
    Yes = khách ĐANG TRẢ TIỀN bị chặn hoàn toàn, ngay lúc này.
    No  = mọi thứ khác: lỗi giao diện, đòi tính năng, câu hỏi, có cách né,
          user free — dù ticket có gào to đến đâu.

Chạy:  ./venv/bin/python -m prompt_tuning_framework.examples.hard_example
"""
import logging

from prompt_tuning_framework import PromptTuner, Sample
from prompt_tuning_framework.components import (AccuracyEvaluator, LLMExecutor,
                                                LLMRewriteOptimizer)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

LABELS = ["Yes", "No"]

SAMPLES = [
    # --- Yes: khách trả tiền bị chặn hoàn toàn (lời lẽ lại rất bình tĩnh) ---
    Sample(id=0, label="Yes",
           text="Enterprise customer cannot log in at all since this morning. "
                "Their entire team is unable to work."),
    Sample(id=1, label="Yes",
           text="Payment API returns 500 for every transaction on the Pro plan. "
                "No orders are going through."),
    Sample(id=2, label="Yes",
           text="A paying customer's production site is down; our database "
                "connector fails on every request."),
    Sample(id=3, label="Yes",
           text="Paid user's dashboard renders a blank page. They cannot reach "
                "any of their data."),

    # --- No: gào to nhưng KHÔNG chặn ai (bẫy chính) ---
    Sample(id=4, label="No",
           text="This is absolutely unacceptable!!! The logo is misaligned on "
                "the settings page."),
    Sample(id=5, label="No",
           text="URGENT!!! I need dark mode immediately, my eyes hurt every night."),
    Sample(id=6, label="No",
           text="CRITICAL: the welcome email has a typo, it says 'Welcom'."),
    Sample(id=7, label="No",
           text="EMERGENCY! Our account manager has not replied to my email "
                "since yesterday afternoon."),

    # --- No: có cách né / chỉ là câu hỏi ---
    Sample(id=8, label="No",
           text="Login is broken in Internet Explorer 11, but it works fine in "
                "Chrome and Firefox."),
    Sample(id=9, label="No",
           text="Free trial user reports the export button takes 8 seconds."),
    Sample(id=10, label="No",
           text="Could you explain how the billing cycle works? I am confused "
                "about the invoice date."),
    Sample(id=11, label="No",
           text="Paying customer would like a CSV export feature added next quarter."),

    # --- No: BỊ CHẶN HOÀN TOÀN nhưng KHÔNG TRẢ TIỀN ---
    # Đây là các ca then chốt. Chúng phá luật "chặn hoàn toàn = khẩn cấp":
    # không có chúng, model chỉ cần học "outage = Yes" là đã đúng hết, nên
    # benchmark không hề kiểm tra được chiều TRẢ TIỀN.
    Sample(id=12, label="No",
           text="Free tier user cannot log in at all and is completely blocked "
                "from using the product."),
    Sample(id=13, label="No",
           text="A trial account's workspace fails to load entirely. The user "
                "cannot access any feature."),
    Sample(id=14, label="No",
           text="Free plan user gets a 500 error on every single API call. "
                "Their integration is completely dead."),

    # --- Yes: trả tiền + bị chặn, lời lẽ vẫn bình tĩnh ---
    Sample(id=15, label="Yes",
           text="Business plan customer cannot reach our API at all; their "
                "integration has been down for an hour."),
]

TASK = ("Classify whether a customer support ticket must be escalated as urgent. "
        "Answer Yes if it is urgent, No if it is not.")

# Prompt mơ hồ: không hề nêu quy định -> sẽ bị các ca gào 'URGENT' đánh lừa
BAD_PROMPT = "Is this support ticket urgent? Yes or No"


def main():
    tuner = PromptTuner(
        # delay=4.5: free tier Gemini chỉ cho 15 request/phút. Không tiết chế thì
        # ca lỗi 429 sẽ bị loại khỏi mẫu số -> điểm bị thổi phồng thành 100 giả.
        executor=LLMExecutor(labels=LABELS, delay=5.0),  # model rẻ — vai "thí sinh"
        evaluator=AccuracyEvaluator(),
        optimizer=LLMRewriteOptimizer(labels=LABELS),    # model mạnh — vai "người sửa prompt"
        task_description=TASK,
        max_iters=4,
        target_score=100.0,
    )

    best = tuner.run(BAD_PROMPT, SAMPLES)
    history = tuner.store.history()
    if best is None:
        print("\nKhông có kết quả đáng tin (xem log lỗi ở trên).")
        return

    print("\n" + "=" * 70)
    print("LỊCH SỬ CÁC PHIÊN BẢN PROMPT")
    print("=" * 70)
    for v in history:
        mark = "  <-- TỐT NHẤT" if best and v.version == best.version else ""
        print(f"\n[v{v.version}] {v.score}/100{mark}\n{v.text}")

    # Chi tiết đúng/sai của prompt tốt nhất
    print("\n" + "=" * 70)
    print("ĐÁNH GIÁ TỪNG CA (prompt tốt nhất)")
    print("=" * 70)
    res = AccuracyEvaluator().evaluate(
        best.text, tuner.executor.execute(best.text, SAMPLES), SAMPLES)
    for r in res.results:
        dau = "OK  " if r.correct else ("BỎ  " if r.correct is None else "SAI ")
        doan = r.predicted if r.correct is not None else "(lỗi gọi LLM)"
        print(f"{dau} đoán={doan:<14} đúng={r.expected:<4} | {r.sample.text[:50]}")
    print(f"\nChấm được {res.num_scored}/{len(SAMPLES)} ca"
          f"{'' if res.reliable else '  <-- KHÔNG ĐÁNG TIN'}")

    print("\n" + "=" * 70)
    print(f"TRƯỚC: {history[0].score}/100  |  SAU: {best.score}/100")
    print("=" * 70)


if __name__ == "__main__":
    main()
