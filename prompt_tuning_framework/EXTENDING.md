# Mở rộng framework

File này dành cho bạn nếu bạn muốn cắm component của mình vào, hoặc muốn hiểu bên
trong framework chạy thế nào. Chỉ dùng framework thì không cần đọc — xem
[README](README.md) là đủ.

---

## Vòng lặp khép kín 4 bước

```
      ┌──────────────────────────────────────────┐
      │            PromptTuner.run()             │
      │       (framework giữ vòng lặp)           │
      └────────────────┬─────────────────────────┘
                       │  mỗi vòng:
   ① BasePromptStore ──┤   store.save() / record_score()   → Quản lý Prompt
   ② BaseExecutor    ──┤   executor.execute(prompt, ...)   → Thực thi
   ③ BaseEvaluator   ──┤   evaluator.evaluate(...)         → Đánh giá
   ④ BaseOptimizer   ──┘   optimizer.propose(errors)       → Tối ưu hóa
```

Điểm mấu chốt: bốn lời gọi đó nằm trong code của framework, không phải code của
bạn. Bạn chỉ đưa component vào rồi gọi `tuner.run()` đúng một lần, còn vòng lặp
thuộc về framework. Người ta gọi cái đó là Inversion of Control, và nó là ranh
giới phân biệt framework với thư viện thường.

Muốn thấy tận mắt:

```bash
grep -n "executor.execute\|evaluator.evaluate\|store.record_score\|optimizer.propose" \
  prompt_tuning_framework/core/tuner.py
```

Bốn dòng cần nhìn là 102, 104, 121, 146 — đều nằm bên trong vòng `for`.

---

## Cắm component của bạn

```python
from prompt_tuning_framework import BaseEvaluator, register

@register("evaluator", "my_eval")          # đăng ký để dùng được trong YAML
class MyEvaluator(BaseEvaluator):
    def evaluate(self, prompt, predictions, samples):
        ...                                 # framework sẽ GỌI hàm này
        return EvalResult(score=..., results=[...])
```

Tương tự với `BasePromptStore`, `BaseExecutor`, `BaseOptimizer`, `BaseCallback`.

Xem những gì đã đăng ký:

```python
from prompt_tuning_framework import available
available("optimizer")   # ['autoprompt', 'llm_rewrite']
```

Ví dụ đầy đủ: `examples/custom_components.py`

---

## Dùng bằng YAML

```python
from prompt_tuning_framework import tuner_from_yaml
tuner = tuner_from_yaml("prompt_tuning_framework/examples/config_example.yml")
best = tuner.run(initial_prompt, samples)
```

Đổi `optimizer.name` giữa `autoprompt` và `llm_rewrite` mà không sửa dòng code
nào.

---

## Cấu trúc thư mục

```
prompt_tuning_framework/
├── core/
│   ├── types.py        Sample, Prediction, EvalResult, PromptVersion
│   ├── stats.py        Wilson CI, Clopper-Pearson, McNemar, non-inferiority
│   ├── interfaces.py   ⭐ 4 điểm mở rộng (abstract)
│   ├── registry.py     Đăng ký plugin theo tên
│   └── tuner.py        ⭐ PromptTuner — giữ vòng lặp
├── components/
│   ├── stores/         InMemoryPromptStore, SQLitePromptStore
│   ├── executors/      LLMExecutor, MultiModelExecutor
│   ├── evaluators/     AccuracyEvaluator, CompositeEvaluator, CrossModelEvaluator
│   └── optimizers/     LLMRewriteOptimizer, AutoPromptOptimizer (adapter)
├── llm.py              Provider + model mặc định + tìm API key
├── config.py           YAML → tự dựng component
├── cli.py              Lệnh `prompt-tune`
├── data.py             Nạp CSV, tách train/test
├── ui/                 UI demo Streamlit (một consumer của framework)
├── tests/              205 test, chạy offline
└── examples/           quickstart, hard_example (ticket), pii_example (tiếng Việt)
```

---

## Chạy test

```bash
pip install -e "prompt_tuning_framework/[test]"
python -m pytest prompt_tuning_framework/tests/ -q
```

205 test, chạy offline, không cần API key. Mọi lời gọi LLM trong test đều bị thay
bằng hàng giả.

Muốn chạy cả phần đối chiếu thống kê với `statsmodels`:

```bash
pip install -e "prompt_tuning_framework/[test,verify]"
python -m pytest prompt_tuning_framework/tests/test_metrics_verification.py -v
```

---

## Quan hệ với AutoPrompt

AutoPrompt chỉ là một plugin optimizer (`AutoPromptOptimizer`), không phải lõi.
Framework chạy hoàn toàn không cần nó — `LLMRewriteOptimizer` là bản mặc định và
tự đủ.

Plugin này cần cả repo AutoPrompt nằm trên đĩa, vì nó `import utils.llm_chain` và
đọc file meta-prompt trực tiếp từ repo. pip không cài được thứ đó: AutoPrompt của
Eladlev không có trên PyPI, còn gói tên `autoprompt` trên PyPI là của tác giả
khác, không liên quan.

```bash
git clone https://github.com/Eladlev/AutoPrompt.git
cd AutoPrompt
pip install -e "prompt_tuning_framework/[autoprompt]"
```

Lưu ý là `[autoprompt]` ghim `langchain<0.3` (theo `requirements.txt` của
upstream, ghim `langchain==0.2.7`). Cái pin đó kéo `langchain-google-genai` từ
4.x tụt về 1.x, tức hạ cấp SDK Gemini về đời 2024. Đó là lý do tôi để nó nằm
ngoài base install lẫn ngoài `[all]`.
