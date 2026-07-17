# Cách dùng

Cài đặt xem [README](README.md). File này nói cách chạy.

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

| Provider | Chạy prompt (executor) | Tối ưu prompt (optimizer) |
|----------|------------------------|---------------------------|
| `google` | `gemini-3.1-flash-lite` | `gemini-3.5-flash` |
| `openai` | `gpt-4o-mini` | `gpt-4o-mini` |

Hai vai này rất khác nhau. Executor gọi rất nhiều lần nhưng mỗi lần bé tí (khoảng
48 token vào, một nhãn ra). Optimizer chỉ gọi một lần mỗi vòng nhưng sinh ra cả
một prompt dài, nên nó mới là phần tốn tiền.

Vậy nên muốn rẻ hơn thì hạ model optimizer, không phải model executor:

```bash
prompt-tune run --optimizer-model gemini-3-flash-preview ...
```

---

## Chạy trên terminal

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

### Tách test set ra, nếu muốn con số có nghĩa

Optimizer được xem các câu trả lời sai để sửa prompt. Nếu rồi lại chấm điểm trên
chính những sample đó thì điểm thu được là điểm học thuộc, không nói lên gì về
sample mới.

```python
from prompt_tuning_framework import split_samples

dev, test = split_samples(samples, test_ratio=0.5, seed=0)
best = tuner.run(prompt, dev, test_samples=test)

print(best.metadata["test_score"])      # con số đáng công bố
```

Vì sao chuyện này quan trọng: xem [MEASUREMENT.md](MEASUREMENT.md).

---

## Chạy thử ví dụ

```bash
# chó/mèo, 4 sample, xong trong vài giây
python -m prompt_tuning_framework.examples.quickstart

# phát hiện lộ thông tin khách hàng, tiếng Việt, 120 sample
python -m prompt_tuning_framework.examples.pii_example

# ticket hỗ trợ, 480 sample, bài khó nhất — khoảng 1.300 VND
python -m prompt_tuning_framework.examples.hard_example --delay 0 --workers 8

# bản rút gọn của bài trên, vài phút
python -m prompt_tuning_framework.examples.hard_example --nho 24 --max-iters 2
```

Kết quả của các bài này: xem [RESULTS.md](RESULTS.md).
