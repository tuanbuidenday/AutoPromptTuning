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

## Cấu trúc

```
prompt_tuning_framework/
├── core/
│   ├── types.py        Sample, Prediction, EvalResult, PromptVersion
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
