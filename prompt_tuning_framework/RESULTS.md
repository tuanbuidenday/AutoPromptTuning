# Kết quả thật

Mọi lần chạy dưới đây đều dùng Gemini thật. Điểm công bố lấy từ **test set** mà
optimizer chưa từng nhìn thấy.

Cách đọc các con số này (khoảng tin cậy là gì, McNemar là gì, vì sao lấy min chứ
không lấy trung bình): xem [MEASUREMENT.md](MEASUREMENT.md).

---

## Bài ticket hỗ trợ (tiếng Anh)

480 sample, chia 280 train / 200 test. Chạy 4 vòng, 1.320 lượt gọi, chấm đủ
200/200 không lỗi.

| | Train | Test |
|---|---|---|
| Prompt gốc | 68.9 | 71.5 — khoảng tin cậy 95% [64.9, 77.3] |
| Prompt tối ưu | 100.0 | **100.0** — [98.1, 100.0] |

McNemar: 57 flip (sai thành đúng), 0 sample xấu đi, p = 1.4 × 10⁻¹⁷.

Train 100 mà test cũng 100, nên không có chuyện học thuộc.

Framework tự suy ra quy định ẩn (khách phải vừa trả tiền vừa bị chặn hoàn toàn)
chỉ từ các câu trả lời sai, và tự vô hiệu hoá bẫy giọng điệu theo cả hai chiều:
prompt nó viết ra nói rõ ticket gào "URGENT" vẫn có thể là No, mà khách lịch sự
viết "no rush" vẫn có thể là Yes.

Chi phí khoảng 1.300 VND một lần chạy đầy đủ. Rẻ nhờ tách vai: executor chạy
1.320 lượt nhưng dùng model rẻ và chỉ trả về một từ, còn optimizer đắt gấp 6 lần
trên mỗi token thì chỉ gọi 3 lượt.

```bash
python -m prompt_tuning_framework.examples.hard_example --delay 0 --workers 8
```

---

## Bài phát hiện lộ thông tin khách hàng (tiếng Việt)

120 sample, chia 60 train / 60 test.

Prompt khởi đầu là kiểu người ta hay viết, không định nghĩa gì:

```
Dữ liệu đầu vào là nhạy cảm, lộ thông tin khách hàng, trả lời Yes or No
```

| | Train | Test |
|---|---|---|
| Prompt gốc | 61.7 | 66.7 — [54.1, 77.3] |
| Prompt tối ưu | 100.0 | **100.0** — [94.0, 100.0] |

McNemar: 20 flip, 0 sample xấu đi, p = 1.9 × 10⁻⁶.

Chỗ đáng chú ý: prompt gốc có chữ "nhạy cảm", mà dataset rải chữ đó đều 50/50
giữa hai nhãn. Framework tự nhận ra chữ đó vô nghĩa và viết thẳng vào prompt rằng
văn bản chỉ nhắc tới "bảo mật" mà không nêu dữ liệu cụ thể thì là No.

Khoảng tin cậy ở đây rộng hơn bài ticket ([94, 100] so với [98.1, 100]) vì chỉ có
60 sample test thay vì 200. Ít sample thì kết luận yếu hơn, không tránh được.

```bash
python -m prompt_tuning_framework.examples.pii_example
```

---

## Plugin AutoPrompt — engine ngoài cắm vào

Train 60 / test 200.

| | Train | Test |
|---|---|---|
| Prompt gốc | 70.0 | — |
| Prompt tối ưu | 86.7 | **85.0** — [79.4, 89.3] |

Cái này không nhằm khoe điểm, mà để chứng minh một engine có sẵn ngoài đời cắm
vào được qua đúng interface `BaseOptimizer`, không sửa dòng nào trong lõi.

> **Đừng so 85.0 với 100.0 ở trên.** Hai lần chạy khác điều kiện — một bên 280
> sample train và 4 vòng, một bên 60 sample — nên chênh lệch có thể hoàn toàn do
> lượng dữ liệu chứ không phải do engine. Muốn biết engine nào hơn thì phải chạy
> lại cùng điều kiện, và bảng này không nhằm trả lời câu đó.

---

## Prompt có kén model không

Đo trên 3 model Gemini, 60 sample.

| | Prompt gốc | Prompt tối ưu |
|---|---|---|
| `gemini-2.5-flash-lite` | 73.3 | 95.0 |
| `gemini-3.1-flash-lite` | 76.7 | 100.0 |
| `gemini-3-flash-preview` | 81.7 | 100.0 |
| **Điểm công bố (min)** | **73.3** | **95.0** |
| Trung bình | 77.2 | 98.3 |
| Chênh lệch (max − min) | 8.4 | 5.0 |

Hai điều đáng rút ra.

Chênh lệch giảm từ 8.4 xuống 5.0, tức prompt tối ưu **ít kén model hơn** prompt
gốc, dù nó chỉ được tinh chỉnh trên đúng một model.

Và lấy min là đúng: trung bình 98.3 che mất việc `gemini-2.5-flash-lite` chỉ được
95.0. Nếu công bố 98.3 thì người dùng model đó sẽ thấy khác hẳn với quảng cáo.

---

## Tổng hợp

| | |
|---|---|
| Test | 205 passed, 3 skipped |
| Bài đã đo bằng LLM thật | 2 miền — ticket tiếng Anh, PII tiếng Việt |
| Optimizer đã kiểm chứng | 2 — `llm_rewrite`, plugin `autoprompt` |
| Chi phí một lần chạy đầy đủ | ~1.300 VND |
