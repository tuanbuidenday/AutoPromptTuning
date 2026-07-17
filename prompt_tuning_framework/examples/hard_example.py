"""Ví dụ KHÓ — phân loại ticket hỗ trợ: có khẩn cấp không?

Vì sao khó: "khẩn cấp" ở đây là QUY ĐỊNH NỘI BỘ của doanh nghiệp, không phải
kiến thức phổ thông. LLM không thể đoán ra, và các ca bẫy cố tình gào "URGENT!!!",
"CRITICAL" để dụ nó trả lời sai. Prompt ban đầu chắc chắn hỏng; framework phải
suy ra được quy định chỉ từ các CA SAI.

Quy định thật (KHÔNG hề nói cho model biết):
    Yes = khách ĐANG TRẢ TIỀN bị chặn hoàn toàn, ngay lúc này.
    No  = mọi thứ khác: lỗi giao diện, đòi tính năng, câu hỏi, có cách né,
          user free — dù ticket có gào to đến đâu.

Bộ mẫu (examples/tickets.csv, sinh bởi make_tickets.py): 480 ca, cân bằng 240 Yes
/ 240 No, chia 280 train / 200 test. Thiết kế giai thừa giọng-điệu × trả-tiền ×
bị-chặn để mọi LUẬT LƯỜI đều thất bại — giọng điệu chỉ đạt 50 điểm (bằng tung
đồng xu), mỗi dấu hiệu đơn lẻ 83.3 điểm. Chỉ hiểu đúng sự KẾT HỢP mới đạt 100.
test_dataset.py canh giữ điều này.

Ví dụ này TÁCH train/test. Optimizer chỉ nhìn thấy tập train; điểm công bố lấy từ
tập test mà nó chưa từng thấy. Không tách thì điểm cao chỉ là điểm học thuộc:
optimizer được xem đúng các ca sai rồi viết prompt vá chúng. Đã kiểm chứng bằng
LLM thật: một lần chạy cho train 83.3 nhưng test chỉ 66.7 — chênh 16.6 điểm.

CHI PHÍ — đọc kỹ trước khi chạy: mỗi vòng gọi LLM một lần cho MỖI ca train (280
ca), cộng 200 lần chấm tập test ở cuối. Chạy đủ 4 vòng là ~1.320 request, khoảng
1.300 VND trên gói trả phí. Free tier chỉ cho 20 request/NGÀY/model (tính theo
PROJECT, không theo key) nên chắc chắn cạn quota giữa chừng — ca lỗi sẽ bị loại
khỏi mẫu số làm điểm bị thổi phồng (cờ reliable sẽ chặn, nhưng bạn mất thời gian).
Dùng --nho để chạy thử trên bộ nhỏ trước.

Chạy:  ./venv/bin/python -m prompt_tuning_framework.examples.hard_example
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

from prompt_tuning_framework import (PromptTuner, Sample, load_samples_csv,
                                     split_samples)
from prompt_tuning_framework.components import (AccuracyEvaluator, LLMExecutor,
                                                LLMRewriteOptimizer)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

LABELS = ["Yes", "No"]
DATASET = Path(__file__).parent / "tickets.csv"

# Bộ 480 ca chia 280 train / 200 test. Con số 200 không tuỳ tiện: đó là cỡ nhỏ
# nhất mà non_inferiority(margin_pp=5) còn kết luận được. Với 120 ca thì chỉ kết
# luận nổi ở mức 7 điểm.
N_TEST = 200

TASK = ("Classify whether a customer support ticket must be escalated as urgent. "
        "Answer Yes if it is urgent, No if it is not.")

# Prompt mơ hồ: không hề nêu quy định -> sẽ bị các ca gào 'URGENT' đánh lừa
BAD_PROMPT = "Is this support ticket urgent? Yes or No"


def main(max_iters: int = 4, n_mau: Optional[int] = None,
         delay: float = 5.0, num_workers: int = 1):
    """:param n_mau: chỉ lấy ngần này ca (cân bằng nhãn) để chạy thử cho rẻ.
        None = dùng cả 480 ca, tốn ~1.320 request cho 4 vòng.
    :param delay: nghỉ bao nhiêu giây giữa 2 lần gọi. Mặc định 5.0 cho free tier
        (15 request/phút). Gói TRẢ PHÍ có RPM cao hơn nhiều -> để 0 và tăng
        num_workers, nếu không 1.320 lượt sẽ mất gần 2 tiếng.
    :param num_workers: số luồng song song. Chỉ tăng khi đã bật billing — bị rate
        limit sẽ làm ca lỗi, mà ca lỗi bị loại khỏi mẫu số nên điểm bị thổi phồng.
    """
    samples = load_samples_csv(str(DATASET))

    if n_mau is not None and n_mau < len(samples):
        # Cắt phải GIỮ CÂN BẰNG NHÃN, không cắt thẳng n ca đầu: bộ lệch nhãn sẽ
        # khiến prompt luôn trả nhãn đa số ăn điểm cao mà không hiểu gì.
        yes = [s for s in samples if s.label == "Yes"][: n_mau // 2]
        no = [s for s in samples if s.label == "No"][: n_mau - n_mau // 2]
        samples = yes + no

    # seed cố định -> chia lại y hệt, để so sánh giữa các lần chạy là công bằng.
    # Đúng 200 ca test là con số có chủ đích: dưới mức đó thì khoảng tin cậy quá
    # rộng để kết luận "prompt mới không tệ hơn 5 điểm" (120 ca chỉ đủ cho 7 điểm).
    if n_mau is not None:
        dev, test = split_samples(samples, test_ratio=0.5, seed=0)
    else:
        dev, test = split_samples(samples, test_size=N_TEST, seed=0)

    print(f"Bộ mẫu: {len(samples)} ca  ->  train {len(dev)} / test {len(test)} "
          f"(test được giữ riêng, optimizer không thấy)")
    print(f"Ước tính: ~{len(dev) * max_iters + len(test)} request LLM")

    tuner = PromptTuner(
        # delay/num_workers: free tier Gemini chỉ cho 15 request/phút. Không tiết
        # chế thì ca lỗi 429 sẽ bị loại khỏi mẫu số -> điểm bị thổi phồng thành
        # 100 giả. Đã bật billing thì để delay=0 và tăng num_workers.
        executor=LLMExecutor(labels=LABELS, delay=delay,   # model rẻ — vai "thí sinh"
                             num_workers=num_workers),
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
        # Nhận xét phải bám số thật, đừng cảnh báo "sai số rộng" khi nó đang hẹp.
        sai_so = (hi - lo) / 2
        if sai_so > 8:
            print(f"Sai số ±{sai_so:.1f} điểm là RẤT RỘNG — bộ test {md['test_num_scored']} ca")
            print("quá nhỏ để kết luận. Điểm trần đang tạo cảm giác chắc chắn không có thật.")
        elif sai_so > 4.5:
            print(f"Sai số ±{sai_so:.1f} điểm: đủ thấy khác biệt lớn, chưa đủ để chứng minh")
            print("hai prompt tương đương trong khoảng 5 điểm.")
        else:
            print(f"Sai số ±{sai_so:.1f} điểm — đủ hẹp để kết luận, kể cả cho câu hỏi")
            print("'prompt ngắn hơn có tệ hơn quá 5 điểm không'.")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nho", type=int, default=None, metavar="N",
                    help="chỉ dùng N ca (cân bằng nhãn) cho rẻ, vd --nho 24")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--delay", type=float, default=5.0,
                    help="giây nghỉ giữa 2 lần gọi (0 nếu đã bật billing)")
    ap.add_argument("--workers", type=int, default=1,
                    help="số luồng song song (chỉ tăng khi đã bật billing)")
    a = ap.parse_args()
    main(max_iters=a.max_iters, n_mau=a.nho, delay=a.delay, num_workers=a.workers)
