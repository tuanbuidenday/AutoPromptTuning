# 🧩 Prompt Tuning Framework

Framework tối ưu hóa & tinh chỉnh prompt tự động / bán tự động.

Framework giữ vòng lặp chính và gọi ngược lại code của bạn (Inversion of Control) —
bạn chỉ cắm component vào 4 điểm mở rộng.

## Cài đặt

```bash
pip install -e "prompt_tuning_framework/[google]"   # Google Gemini
pip install -e "prompt_tuning_framework/[openai]"   # OpenAI
pip install -e "prompt_tuning_framework/[all]"      # tất cả + AutoPrompt + test
```

## Gắn API key

```bash
export GOOGLE_API_KEY="..."      # --provider google (mặc định)
export OPENAI_API_KEY="sk-..."   # --provider openai
```

Hoặc lưu vào `config/llm_env.local.yml` (đã được `.gitignore`):

```yaml
google:
    GOOGLE_API_KEY: '...'
openai:
    OPENAI_API_KEY: 'sk-...'
```

Thứ tự ưu tiên: tham số `api_key=` → biến môi trường → `llm_env.local.yml` → `llm_env.yml`.

Điền key vào `llm_env.local.yml`, không phải `llm_env.yml` — file sau nằm trong Git.
Không có cờ `--api-key`; dùng biến môi trường.

## Chọn model

Không chỉ định thì framework lấy model rẻ nhất của provider:

| Provider | Chạy prompt | Tối ưu prompt |
|----------|-------------|---------------|
| `google` | `gemini-3.1-flash-lite` | `gemini-3.5-flash` |
| `openai` | `gpt-4o-mini` | `gpt-4o-mini` |

Model *chạy prompt* gọi nhiều lần nhưng mỗi lần rất nhỏ (~48 token vào, 1 nhãn ra).
Model *tối ưu prompt* chỉ gọi 1 lần mỗi vòng nhưng sinh ra prompt dài — nên nó mới là
phần **tốn nhất**: với 16 ca × 2 vòng, optimizer chiếm ~83% chi phí (~71 VND/lần chạy).

Muốn rẻ hơn thì hạ model tối ưu, không phải model chạy:

```bash
prompt-tune run --optimizer-model gemini-3-flash-preview ...   # rẻ hơn ~2 lần
```

## Dùng trên terminal

```bash
prompt-tune plugins                    # liệt kê plugin đã đăng ký

prompt-tune run \
    --dataset data.csv \
    --prompt "Is this a dog? Yes or No" \
    --task "Classify dog vs cat. Yes = dog, No = cat." \
    --labels Yes No \
    --max-iters 3

prompt-tune run --provider openai --dataset data.csv --prompt "..." --labels Yes No
prompt-tune run --config my_config.yml --dataset data.csv --prompt "..."
```

`data.csv` cần 2 cột `text,label`:

```csv
text,label
It barks at strangers.,Yes
It purrs on my lap.,No
```

Output:

```
Nạp 4 ca test (4 ca có nhãn) từ data.csv
  vòng 0:  50.0/100  (2/4 đúng, 2 sai)
  vòng 1: 100.0/100  (4/4 đúng, 0 sai)

TRƯỚC: (50.0/100) Is this a dog? Yes or No
SAU  : (100.0/100) A dog barks or walks on a leash; a cat purrs...
```

## Chạy test

```bash
pip install -e "prompt_tuning_framework/[test]"
python -m pytest prompt_tuning_framework/tests/ -q
```

89 test, chạy offline, không cần API key.

## UI demo

```bash
pip install streamlit
streamlit run prompt_tuning_framework/ui/streamlit_app.py
```

## Python API

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

## Dùng bằng YAML

```python
from prompt_tuning_framework import tuner_from_yaml
tuner = tuner_from_yaml("prompt_tuning_framework/examples/config_example.yml")
best = tuner.run(initial_prompt, samples)
```

Đổi `optimizer.name` giữa `autoprompt` ⇄ `llm_rewrite` mà không sửa dòng code nào.

## Mở rộng — cắm component của bạn

```python
from prompt_tuning_framework import BaseEvaluator, register

@register("evaluator", "my_eval")          # đăng ký để dùng được trong YAML
class MyEvaluator(BaseEvaluator):
    def evaluate(self, prompt, predictions, samples):
        ...                                 # framework sẽ GỌI hàm này
        return EvalResult(score=..., results=[...])
```

Tương tự với `BasePromptStore`, `BaseExecutor`, `BaseOptimizer`, `BaseCallback`.

```python
from prompt_tuning_framework import available
available("optimizer")   # ['autoprompt', 'llm_rewrite']
```

## Vòng lặp khép kín 4 bước

```
      ┌──────────────────────────────────────────┐
      │            PromptTuner.run()             │
      │  (framework giữ vòng lặp — IoC)          │
      └────────────────┬─────────────────────────┘
                       │  mỗi vòng:
   ① BasePromptStore ──┤   store.save() / record_score()   → Quản lý Prompt
   ② BaseExecutor    ──┤   executor.execute(prompt, ...)   → Thực thi
   ③ BaseEvaluator   ──┤   evaluator.evaluate(...)         → Đánh giá
   ④ BaseOptimizer   ──┘   optimizer.propose(errors)       → Tối ưu hóa
```

## Đo lường hiệu quả prompt

Một điểm accuracy trần là con số gây hiểu nhầm. Framework cung cấp sẵn các công
cụ để đo cho đúng.

### Bộ mẫu chuẩn — thiết kế để luật lười thất bại

`examples/tickets.csv`: **480 ca, cân bằng 240 Yes / 240 No**, chia **280 train /
200 test**. Sinh bởi `examples/make_tickets.py` (seed cố định, tái tạo được).

Quy định ẩn: **Yes ⟺ khách đang trả tiền VÀ bị chặn hoàn toàn.** Bộ mẫu theo
thiết kế giai thừa *giọng điệu × trả tiền × bị chặn*, nên mọi dấu hiệu đơn lẻ đều
không đủ:

| Luật lười | Điểm đạt được |
|---|---|
| "gào URGENT!!! → Yes" (giọng điệu) | **50.0** — vô dụng, bằng tung đồng xu |
| "bình tĩnh → Yes" (giọng điệu) | **50.0** — vô dụng |
| "khách trả tiền → Yes" | 83.3 — chưa đủ |
| "bị chặn → Yes" | 83.3 — chưa đủ |
| **"trả tiền VÀ bị chặn → Yes"** | **100** — luật thật |

Ticket gào to rải đều cả hai nhãn (`P(Yes | gào to) = 50.0%`). Điều này quan
trọng: nếu chỉ ticket No mới gào to thì model chỉ cần học "gào to → No", và
benchmark sẽ đo **giọng điệu** thay vì đo quy định. `tests/test_bo_mau.py` canh
giữ toàn bộ các tính chất này.

Cỡ tập test = 200 không tuỳ tiện — đó là cỡ nhỏ nhất đủ cho mục tiêu "rút ngắn
prompt mà accuracy không tụt quá 5 điểm":

| Cỡ tập test | Khoảng tin cậy (ở 90%) | Kết luận non-inferiority 5đ? |
|---|---|---|
| 16 | ±16.3 | chưa đủ |
| 48 | ±8.8 | chưa đủ |
| 120 | ±5.4 | chưa đủ (chỉ tới 7đ) |
| **200** | **±4.2** | **được** |

```python
dev, test = split_samples(samples, test_size=200, seed=0)   # 280 / 200
```

Sinh lại: `python -m prompt_tuning_framework.examples.make_tickets`

### Tách tập test — bắt buộc nếu muốn con số có nghĩa

Optimizer được xem các ca **sai** để viết lại prompt. Nếu chấm điểm trên chính
những ca đó thì prompt chỉ đang vá thuộc lòng, và 100/100 thu được là **điểm học
thuộc**, không nói lên gì về ca mới.

```python
from prompt_tuning_framework import split_samples

dev, test = split_samples(samples, test_ratio=0.5, seed=0)  # phân tầng theo nhãn
best = tuner.run(prompt, dev, test_samples=test)            # optimizer không thấy test

print(best.metadata["test_score"])                          # con số đáng công bố
print(best.metadata["test_ci_low"], best.metadata["test_ci_high"])
```

Điểm dev cao hơn điểm test quá 10 điểm sẽ bị log cảnh báo `HỌC THUỘC`.

### Khoảng tin cậy — luôn đọc kèm điểm

```python
result.score                 # 100.0
result.confidence_interval   # (80.6, 100.0)  <- 16 mẫu chỉ chứng minh được >= ~81%
result.margin_of_error       # ~9.7
```

### Hai prompt có thật sự khác nhau không?

```python
a.distinguishable_from(b)    # kiểm định ghép cặp McNemar trên cùng bộ mẫu
```

Trên 16 mẫu, 100.0 và 93.8 chỉ hơn nhau **đúng một ca** — không phân biệt được.
Cần **6 ca** lật từ sai sang đúng mới đạt p < 0.05:

```python
from prompt_tuning_framework import min_flips_for_significance
min_flips_for_significance(0.05)   # 6
```

### Rút gọn prompt mà không mất độ chính xác

Đây là bài toán **non-inferiority** — chứng minh accuracy KHÔNG tụt, khó hơn
chứng minh nó tăng. Không được suy `p > 0.05` ⇒ "bằng nhau": với bộ mẫu nhỏ thì
p luôn > 0.05, nên lập luận đó sẽ *luôn* kết luận "không đổi" kể cả khi prompt
mới tệ đi thật.

```python
from prompt_tuning_framework import non_inferiority

non_inferiority(base_correct=15, new_correct=15, num_total=16, margin_pp=5.0)
# False — điểm y hệt nhau, nhưng 16 mẫu KHÔNG đủ để kết luận
non_inferiority(base_correct=180, new_correct=180, num_total=200, margin_pp=5.0)
# True
```

### Tối ưu vừa đúng vừa ngắn

```python
from prompt_tuning_framework.components import CompositeEvaluator

evaluator = CompositeEvaluator(word_budget=50, brevity_weight=10)
# điểm = accuracy - brevity_weight * phần_trăm_vượt_ngân_sách
```

Chỉ phạt khi prompt **dài hơn** ngân sách; ngắn hơn không được thưởng — nếu
thưởng, optimizer sẽ cắt prompt tới mức cụt lủn để ăn điểm.

### Prompt có tốt cho nhiều model không?

```python
from prompt_tuning_framework.components import (MultiModelExecutor,
                                                CrossModelEvaluator)

executor = MultiModelExecutor(models=[
    {"provider": "google", "model": "gemini-3.1-flash-lite"},
    {"provider": "openai", "model": "gpt-4o-mini"},
], labels=LABELS)
evaluator = CrossModelEvaluator()      # điểm = accuracy của model YẾU NHẤT
```

Lấy **min** chứ không lấy trung bình: prompt đạt 100 trên model A và 60 trên
model B có trung bình 80 — nghe ổn, nhưng đó không phải prompt dùng chung được.
`metrics["accuracy_spread"]` cho biết prompt kén model tới mức nào.

## Cấu trúc

```
prompt_tuning_framework/
├── core/
│   ├── types.py        Sample, Prediction, EvalResult, PromptVersion
│   ├── stats.py        Wilson CI, McNemar, non-inferiority
│   ├── interfaces.py   ⭐ 4 điểm mở rộng (abstract)
│   ├── registry.py     Đăng ký plugin theo tên
│   └── tuner.py        ⭐ PromptTuner — giữ vòng lặp (IoC)
├── components/
│   ├── stores/         InMemoryPromptStore, SQLitePromptStore
│   ├── executors/      LLMExecutor (Google / OpenAI)
│   ├── evaluators/     AccuracyEvaluator
│   └── optimizers/     LLMRewriteOptimizer, AutoPromptOptimizer (adapter)
├── llm.py              Provider + model mặc định + tìm API key
├── config.py           YAML → tự dựng component
├── cli.py              Lệnh `prompt-tune`
├── ui/                 UI demo Streamlit (một consumer của framework)
├── tests/              89 test, chạy offline
└── examples/           quickstart.py, hard_example.py, custom_components.py
```

## Chạy ví dụ

```bash
./venv/bin/python -m prompt_tuning_framework.examples.quickstart          # chó/mèo, đơn giản
./venv/bin/python -m prompt_tuning_framework.examples.hard_example        # ticket hỗ trợ, khó
./venv/bin/python -m prompt_tuning_framework.examples.custom_components   # tự cắm component
```

`hard_example.py` — phân loại ticket hỗ trợ theo một quy định nội bộ mà LLM không thể
đoán ra, và các ca bẫy cố tình gào "URGENT!!!". Chạy thật với Gemini: **68.8 → 100/100**
(16/16 ca), framework tự suy ra quy định chỉ từ các ca đoán sai.

## Quan hệ với AutoPrompt

AutoPrompt chỉ là một plugin optimizer (`AutoPromptOptimizer`), không phải lõi.
Framework chạy được hoàn toàn không cần AutoPrompt — xem `LLMRewriteOptimizer`.
