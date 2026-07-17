# 🧩 Prompt Tuning Framework

Framework tối ưu và tinh chỉnh prompt, tự động hoặc bán tự động.

Bạn viết một prompt, đưa cho nó dataset có đáp án, rồi nó tự chạy vòng lặp: chấm
điểm, nhặt ra các câu trả lời sai, nhờ một model mạnh viết lại prompt, rồi lặp
lại. Cuối cùng bạn nhận prompt tốt nhất, kèm bằng chứng thống kê rằng nó thật sự
tốt hơn chứ không phải bạn tưởng thế.

Nó là framework chứ không phải công cụ, vì vòng lặp thuộc về nó chứ không thuộc
về bạn. Bạn chỉ cắm component vào bốn điểm mở rộng và nó gọi ngược lại.

---

## Cài đặt

Cần Python 3.10 trở lên.

```bash
pip install "prompt-tuning-framework @ git+https://github.com/tuanbuidenday/AutoPromptTuning.git#subdirectory=prompt_tuning_framework"
```

Không cần thêm gì cả — **base install** (bản cài không kèm extras) đã có sẵn cả
Gemini lẫn OpenAI. Cài xong bạn kiểm tra được ngay, chưa cần API key:

```bash
prompt-tune plugins
```

Nó in ra danh sách plugin là xong.

### Extras (tuỳ chọn)

| Extras | Thêm gì | Khi nào cần |
|---|---|---|
| `[test]` | pytest | Chạy bộ test |
| `[verify]` | statsmodels, scipy, numpy | Đối chiếu phần thống kê với thư viện độc lập |
| `[all]` | cả hai cái trên | Bạn muốn đủ đồ nghề để phát triển |
| `[autoprompt]` | easydict, langchain <0.3 | Chỉ khi bạn dùng plugin optimizer AutoPrompt |

Tôi cố ý để `[autoprompt]` đứng ngoài base install và ngoài `[all]`, vì nó ghim
`langchain<0.3` và sẽ kéo tụt SDK Gemini về đời 2024 cho cả người không dùng tới
nó. Lý do đầy đủ ở [EXTENDING.md](EXTENDING.md).

> Nếu dùng cú pháp `pip install "tên_gói[extras] @ git+..."` thì extras phải đứng
> trước dấu `@` theo chuẩn PEP 508. Viết thành
> `...#subdirectory=prompt_tuning_framework[test]` thì pip coi `[test]` là một
> phần của tên thư mục và lặng lẽ bỏ qua extras — cài xong mà thiếu, không báo gì.

---

## Tài liệu

| File | Nội dung |
|---|---|
| [USAGE.md](USAGE.md) | Gắn API key, chọn model, chạy trên terminal và trong Python |
| [RESULTS.md](RESULTS.md) | Kết quả thật trên Gemini — 3 bài, kèm khoảng tin cậy và kiểm định |
| [MEASUREMENT.md](MEASUREMENT.md) | Làm sao biết prompt thật sự tốt hơn — công thức, dataset, cách kiểm chứng |
| [EXTENDING.md](EXTENDING.md) | Cắm component của bạn, cấu trúc bên trong, chạy test |
| [Q&A.md](Q&A.md) | Câu hỏi thường gặp, mỗi câu kèm lệnh chạy được ngay để tự kiểm chứng |
