# 🧩 Prompt Tuning Framework

Framework tối ưu và tinh chỉnh prompt, tự động hoặc bán tự động.

Bạn viết một prompt, đưa cho nó bộ dữ liệu có đáp án, rồi nó tự chạy vòng lặp:
chấm điểm, nhặt ra các câu trả lời sai, nhờ một model mạnh viết lại prompt, rồi
lặp lại. Cuối cùng bạn nhận prompt tốt nhất, kèm bằng chứng thống kê rằng nó thật
sự tốt hơn chứ không phải bạn tưởng thế.

Nó là framework chứ không phải công cụ, vì vòng lặp thuộc về nó chứ không thuộc
về bạn. Bạn chỉ cắm component vào bốn điểm mở rộng và nó gọi ngược lại.

| Đọc thêm | |
|---|---|
| [Q&A.md](Q&A.md) | Những câu hay bị hỏi, mỗi câu kèm một lệnh chạy được ngay để tự kiểm chứng |
| [DO_LUONG.md](DO_LUONG.md) | Làm sao biết prompt có thật sự tốt hơn — công thức, bộ mẫu, cách kiểm chứng |
| [MO_RONG.md](MO_RONG.md) | Cắm component của bạn, cấu trúc bên trong, chạy test |

---

## Cài đặt

Cần Python 3.10 trở lên.

Chỉ muốn dùng framework trong dự án của mình:

```bash
pip install "prompt-tuning-framework @ git+https://github.com/tuanbuidenday/AutoPromptTuning.git#subdirectory=prompt_tuning_framework"
```

Muốn đọc code, sửa code, hoặc chạy test thì clone về:

```bash
git clone https://github.com/tuanbuidenday/AutoPromptTuning.git
cd AutoPromptTuning
pip install -e "prompt_tuning_framework/"
```

Không cần thêm gì cả — bản cài trần đã có sẵn cả Gemini lẫn OpenAI. Cài xong kiểm
tra ngay, chưa cần API key:

```bash
prompt-tune plugins
```

Nếu nó in ra danh sách plugin thì xong.

### Extras (tuỳ chọn)

| Extras | Thêm gì | Khi nào cần |
|---|---|---|
| `[test]` | pytest | Chạy bộ test |
| `[verify]` | statsmodels, scipy, numpy | Đối chiếu phần thống kê với thư viện độc lập |
| `[all]` | cả hai cái trên | Muốn đủ đồ nghề phát triển |
| `[autoprompt]` | easydict, langchain <0.3 | Chỉ khi dùng plugin optimizer AutoPrompt |

`[autoprompt]` cố ý đứng ngoài bản cài trần và ngoài `[all]`. Lý do đầy đủ ở
[MO_RONG.md](MO_RONG.md), tóm tắt là nó ghim `langchain<0.3` và sẽ kéo tụt SDK
Gemini về đời 2024 cho cả người không dùng tới nó.

> Nếu dùng cú pháp `pip install "tên_gói[extras] @ git+..."` thì extras phải đứng
> trước dấu `@` theo chuẩn PEP 508. Viết thành
> `...#subdirectory=prompt_tuning_framework[test]` thì pip coi `[test]` là một
> phần của tên thư mục và lặng lẽ bỏ qua extras — cài xong mà thiếu, không báo gì.

---

## Gắn API key

```bash
export GOOGLE_API_KEY="..."      # provider google (mặc định)
export OPENAI_API_KEY="sk-..."   # provider openai
```

Hoặc lưu vào `config/llm_env.local.yml` (file này đã được `.gitignore`):

```yaml
google:
    GOOGLE_API_KEY: '...'
openai:
    OPENAI_API_KEY: 'sk-...'
```

Thứ tự ưu tiên: tham số `api_key=` → biến môi trường → `llm_env.local.yml` →
`llm_env.yml`.

Điền key vào `llm_env.local.yml`, đừng điền vào `llm_env.yml` — file sau nằm
trong Git. Cũng không có cờ `--api-key`, vì key nằm trong dòng lệnh sẽ lộ ra
`ps` và lịch sử shell.

---

## Chọn model

Không chỉ định gì thì framework lấy model rẻ nhất của provider:

| Provider | Chạy prompt | Tối ưu prompt |
|----------|-------------|---------------|
| `google` | `gemini-3.1-flash-lite` | `gemini-3.5-flash` |
| `openai` | `gpt-4o-mini` | `gpt-4o-mini` |

Hai vai này rất khác nhau. Model *chạy prompt* gọi rất nhiều lần nhưng mỗi lần bé
tí (khoảng 48 token vào, một nhãn ra). Model *tối ưu prompt* chỉ gọi một lần mỗi
vòng nhưng sinh ra cả một prompt dài, nên nó mới là phần tốn tiền.

Vậy nên muốn rẻ hơn thì hạ model tối ưu, không phải model chạy:

```bash
prompt-tune run --optimizer-model gemini-3-flash-preview ...
```

---

## Dùng trên terminal

```bash
prompt-tune plugins                    # liệt kê plugin đã đăng ký

prompt-tune run \
    --dataset data.csv \
    --prompt "Is this a dog? Yes or No" \
    --task "Classify dog vs cat. Yes = dog, No = cat." \
    --labels Yes No \
    --max-iters 3
```

`data.csv` cần đúng 2 cột `text,label`:

```csv
text,label
It barks at strangers.,Yes
It purrs on my lap.,No
```

Nó in ra:

```
Nạp 4 ca test (4 ca có nhãn) từ data.csv
  vòng 0:  50.0/100  (2/4 đúng, 2 sai)
  vòng 1: 100.0/100  (4/4 đúng, 0 sai)

TRƯỚC: (50.0/100) Is this a dog? Yes or No
SAU  : (100.0/100) A dog barks or walks on a leash; a cat purrs...
```

Đổi provider hoặc dùng file cấu hình:

```bash
prompt-tune run --provider openai --dataset data.csv --prompt "..." --labels Yes No
prompt-tune run --config my_config.yml --dataset data.csv --prompt "..."
```

---

## Dùng trong Python

```python
from prompt_tuning_framework import PromptTuner, Sample
from prompt_tuning_framework.components import (
    LLMExecutor, AccuracyEvaluator, LLMRewriteOptimizer)

samples = [
    Sample(id=0, text="It barks at strangers.", label="Yes"),
    Sample(id=1, text="It purrs on my lap.",    label="No"),
]

tuner = PromptTuner(
    executor=LLMExecutor(labels=["Yes", "No"]),
    evaluator=AccuracyEvaluator(),
    optimizer=LLMRewriteOptimizer(labels=["Yes", "No"]),
    task_description="Classify dog vs cat. Yes = dog, No = cat.",
    max_iters=3,
)
best = tuner.run("Is this a dog? Yes or No", samples)
print(best.text, best.score)
```

Muốn con số có nghĩa thì tách tập test ra, để optimizer không nhìn thấy:

```python
from prompt_tuning_framework import split_samples

dev, test = split_samples(samples, test_ratio=0.5, seed=0)
best = tuner.run(prompt, dev, test_samples=test)

print(best.metadata["test_score"])      # con số đáng công bố
```

Vì sao chuyện này quan trọng: xem [DO_LUONG.md](DO_LUONG.md).

---

## Chạy thử ví dụ

```bash
# chó/mèo, 4 ca, chạy trong vài giây
python -m prompt_tuning_framework.examples.quickstart

# phát hiện lộ thông tin khách hàng, tiếng Việt, 120 ca
python -m prompt_tuning_framework.examples.pii_example

# ticket hỗ trợ, 480 ca, bài khó nhất — khoảng 1.300 VND
python -m prompt_tuning_framework.examples.hard_example --delay 0 --workers 8

# bản rút gọn của bài trên, vài phút
python -m prompt_tuning_framework.examples.hard_example --nho 24 --max-iters 2
```

---

## Kết quả thật

Ba lần chạy dưới đây đều dùng Gemini thật, và điểm công bố lấy từ **tập test** mà
optimizer chưa từng nhìn thấy.

### Bài ticket hỗ trợ (tiếng Anh) — 480 ca, 280 train / 200 test

| | Train | Test |
|---|---|---|
| Prompt gốc | 68.9 | 71.5 — khoảng tin cậy 95% [64.9, 77.3] |
| Prompt tối ưu | 100.0 | **100.0** — [98.1, 100.0] |

Kiểm định McNemar: 57 ca chuyển từ sai sang đúng, không ca nào xấu đi,
p = 1.4 × 10⁻¹⁷. Train 100 mà test cũng 100, nên không có chuyện học thuộc.

Framework tự suy ra quy định ẩn (khách phải vừa trả tiền vừa bị chặn hoàn toàn)
chỉ từ các ca sai, và tự vô hiệu hoá bẫy giọng điệu theo cả hai chiều: prompt nó
viết ra nói rõ ticket gào "URGENT" vẫn có thể là No, mà khách lịch sự viết "no
rush" vẫn có thể là Yes.

Chi phí khoảng 1.300 VND một lần chạy đầy đủ (1.320 lượt gọi).

### Bài phát hiện lộ thông tin khách hàng (tiếng Việt) — 120 ca, 60/60

Prompt khởi đầu là kiểu người ta hay viết, không định nghĩa gì:

```
Dữ liệu đầu vào là nhạy cảm, lộ thông tin khách hàng, trả lời Yes or No
```

| | Train | Test |
|---|---|---|
| Prompt gốc | 61.7 | 66.7 — [54.1, 77.3] |
| Prompt tối ưu | 100.0 | **100.0** — [94.0, 100.0] |

McNemar: 20 ca lật, 0 ca xấu đi, p = 1.9 × 10⁻⁶.

Chỗ hay: prompt gốc có chữ "nhạy cảm", mà bộ mẫu rải chữ đó đều 50/50 giữa hai
nhãn. Framework tự nhận ra chữ đó vô nghĩa và viết thẳng vào prompt rằng văn bản
chỉ nhắc tới "bảo mật" mà không nêu dữ liệu cụ thể thì là No.

Khoảng tin cậy ở đây rộng hơn bài ticket ([94, 100] so với [98.1, 100]) vì chỉ có
60 ca test thay vì 200. Ít mẫu thì kết luận yếu hơn, không tránh được.

### Plugin AutoPrompt — engine ngoài cắm vào

| | Train (60) | Test (200) |
|---|---|---|
| Prompt gốc | 70.0 | — |
| Prompt tối ưu | 86.7 | **85.0** — [79.4, 89.3] |

Cái này không nhằm khoe điểm, mà để chứng minh một engine có sẵn ngoài đời cắm
vào được qua đúng interface `BaseOptimizer`, không sửa dòng nào trong lõi.

> Đừng so 85.0 với 100.0 ở trên. Hai lần chạy khác điều kiện — một bên 280 mẫu
> train và 4 vòng, một bên 60 mẫu — nên chênh lệch có thể hoàn toàn do lượng dữ
> liệu chứ không phải do engine. Muốn biết engine nào hơn thì phải chạy lại cùng
> điều kiện, và bảng này không nhằm trả lời câu đó.

### Prompt có kén model không

Đo trên 3 model Gemini, 60 ca:

| | Prompt gốc | Prompt tối ưu |
|---|---|---|
| Model yếu nhất | 73.3 | 95.0 |
| Trung bình | 77.2 | 98.3 |
| Chênh lệch giữa các model | 8.4 | 5.0 |

Độ chênh giảm từ 8.4 xuống 5.0, tức prompt tối ưu ít kén model hơn prompt gốc,
dù nó chỉ được tinh chỉnh trên đúng một model.

Điểm công bố lấy của model **yếu nhất** chứ không lấy trung bình. Lý do ở
[DO_LUONG.md](DO_LUONG.md).
