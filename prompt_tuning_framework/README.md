# 🧩 Prompt Tuning Framework

Framework tối ưu hóa & tinh chỉnh prompt tự động / bán tự động.

Framework giữ vòng lặp chính và gọi ngược lại code của bạn (Inversion of Control) —
bạn chỉ cắm component vào 4 điểm mở rộng.

> 📋 **[HOI_DAP.md](HOI_DAP.md)** — các câu hỏi thường gặp, mỗi câu kèm **lệnh
> chạy được ngay** để tự kiểm chứng thay vì tin lời tác giả.

## Cài đặt

Cần Python >= 3.10. Chỉ dùng framework, không cần sửa code:

```bash
pip install "prompt-tuning-framework @ git+https://github.com/tuanbuidenday/AutoPromptTuning.git#subdirectory=prompt_tuning_framework"
```

Muốn đọc/sửa code hoặc chạy test thì clone về:

```bash
git clone https://github.com/tuanbuidenday/AutoPromptTuning.git
cd AutoPromptTuning
pip install -e "prompt_tuning_framework/"
```

Không cần extras: bản cài trần đã có sẵn **cả Gemini lẫn OpenAI**. Cài xong kiểm
tra ngay, chưa cần API key:

```bash
prompt-tune plugins
```

Extras chỉ dành cho việc thêm, không phải để chạy được:

| Extras | Thêm gì | Khi nào cần |
|---|---|---|
| `[test]` | pytest | Chạy bộ test |
| `[verify]` | statsmodels, scipy, numpy | Đối chiếu phần thống kê với thư viện độc lập |
| `[all]` | `[test]` + `[verify]` | Muốn đủ đồ nghề phát triển |
| `[autoprompt]` | easydict, langchain <0.3 | Dùng plugin optimizer AutoPrompt — **cài riêng**, xem dưới |

`[autoprompt]` cố ý **không** nằm trong bản cài trần lẫn trong `[all]`, vì hai lý do:

Thứ nhất, nó ghim `langchain<0.3` (theo upstream AutoPrompt, `requirements.txt`
ghim `langchain==0.2.7`). Cái pin đó ép `langchain-core<0.3`, kéo
`langchain-google-genai` từ 4.x tụt về 1.x — tức **âm thầm hạ cấp SDK Gemini về
đời 2024** cho cả người không hề dùng AutoPrompt. Đo thật: gộp nó vào `[all]`
khiến pip phải dò ngược 45 phiên bản `langchain-core` và chạy hơn 9 phút chưa
xong; tách ra thì `[all]` cài trong **21 giây**.

Thứ hai, plugin còn `import utils.llm_chain`, tức cần cả **repo AutoPrompt trong
PYTHONPATH** — pip không cấp được, nên extras này vốn không tự đủ. Muốn dùng thì
cài riêng, trong môi trường của repo AutoPrompt:

```bash
pip install -e "prompt_tuning_framework/[autoprompt]"
```

Nếu dùng cú pháp `pip install "tên_gói[extras] @ git+..."` thì extras phải đứng
trước dấu `@` theo PEP 508. Viết `...#subdirectory=prompt_tuning_framework[test]`
thì pip coi `[test]` là một phần tên thư mục và **lặng lẽ bỏ qua extras**.

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

188 test, chạy offline, không cần API key.

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

## Kết quả thật

Chạy `examples/hard_example.py` với Gemini thật, bộ 480 ca (280 train / 200 test),
4 vòng, 1.320 lượt gọi, chấm đủ 200/200 không lỗi:

| | Train (280) | **Test (200, optimizer chưa từng thấy)** |
|---|---|---|
| Prompt gốc | 68.9 | **71.5** — CI 95% [64.9, 77.3] |
| Prompt tối ưu | 100.0 | **100.0** — CI 95% [98.1, 100.0] |

Kiểm định ghép cặp McNemar trên tập test: **57 ca lật sai→đúng, 0 ca xấu đi,
p = 1.4 × 10⁻¹⁷**. Train 100 = test 100 → không học thuộc.

Framework tự suy ra quy định ẩn (khách trả tiền **và** bị chặn hoàn toàn) chỉ từ
các ca sai, và tự vô hiệu hoá bẫy giọng điệu theo **cả hai chiều** — prompt nó
viết ra nói rõ rằng ticket gào "URGENT" vẫn có thể là No, *và* khách lịch sự viết
"no rush" vẫn có thể là Yes.

Chi phí: ~1.300 VND một lần chạy đầy đủ. Rẻ nhờ tách vai executor/optimizer —
executor chạy 1.320 lượt nhưng dùng model rẻ và chỉ trả một từ, còn optimizer đắt
gấp 6 lần trên mỗi token thì chỉ gọi 3 lượt.

## Phương pháp đo lường được tính thế nào

Toàn bộ công thức nằm ở [`core/stats.py`](core/stats.py), không phụ thuộc
scipy/numpy — chỉ dùng `math` và `itertools` của thư viện chuẩn.

### 1. Độ chính xác + cờ đáng tin

```
điểm      = số_ca_đúng / số_ca_CHẤM_ĐƯỢC × 100
đáng tin  = số_ca_chấm_được ≥ min_scored_ratio × tổng_số_ca      (mặc định 0.8)
```

Ca gọi LLM lỗi bị đánh `correct=None` và **rơi khỏi mẫu số**. Đó là lý do phải có
cờ `reliable`: 9/12 ca lỗi + 3 ca đúng = **100/100** dù prompt rất dở. Khi
`reliable=False`, `PromptTuner` **từ chối ghi nhận điểm** và dừng, thay vì tuyên
bố thắng lợi trên rác.

### Làm sao chắc chắn các phép đo này đúng?

Đừng tin. Phần thống kê do chính tác giả framework tự viết, và tự kiểm chứng công
thức bằng chính công thức đó là vô giá trị — hiểu sai từ đầu thì cả hai lần đều
sai giống hệt nhau. [`tests/test_metrics_verification.py`](tests/test_metrics_verification.py)
kiểm chứng bằng **ba tầng độc lập**, và CI chạy lại mỗi lần push:

| Tầng | Cách làm | Bắt được lỗi gì |
|---|---|---|
| **1. Đối chiếu ngoài** | So với `statsmodels` — cài đặt độc lập, do người khác viết | Cài sai công thức |
| **2. Giá trị tham chiếu** | Khoá cứng giá trị suy được bằng tay (vd `b=0,c=k ⇒ p = 2/2ᵏ`) | Cài sai, kể cả khi không có statsmodels |
| **3. Mô phỏng** | Kiểm tra **tính chất**, không kiểm tra công thức | **Dùng sai công cụ** — tầng 1 và 2 bỏ lọt |

Tầng 3 là mạnh nhất vì nó không dựa vào công thức nào: khoảng tin cậy 95% **phải**
chứa giá trị thật ~95% số lần; kiểm định mức 5% **chỉ được** báo động sai ≤ 5% số
lần. Kèm cả chiều ngược lại — McNemar phải **bắt được** khác biệt thật > 95% số
lần, nếu không thì một hàm luôn trả `p = 1.0` cũng qua được test báo-động-giả.

Kết quả: Wilson và McNemar khớp `statsmodels` **sai lệch 0.000000**;
Clopper-Pearson tự viết (dò nhị phân trên CDF nhị thức) khớp bản của statsmodels
(phân vị Beta qua scipy) tới **10⁻¹⁴** — hai đường tính hoàn toàn khác nhau.

```bash
pip install -e "prompt_tuning_framework/[test,verify]"
pytest prompt_tuning_framework/tests/test_metrics_verification.py -v
```

`statsmodels` **không phải** dependency của framework — `core/stats.py` chỉ dùng
thư viện chuẩn. Nó chỉ cần khi muốn *chứng minh* phép đo đúng.

### 2. Khoảng tin cậy Wilson (95%, z = 1.96)

```
tâm        = (x + z²/2) / (n + z²)
nửa_rộng   = z/(n + z²) × √( x(n−x)/n + z²/4 )
CI         = [tâm − nửa_rộng,  tâm + nửa_rộng]        x = số đúng, n = số chấm được
```

Dùng Wilson **thay vì Wald** (`p ± z√(p(1−p)/n)`) vì Wald vỡ ở rìa: với 16/16 ca
đúng, Wald cho `[100, 100]` — tuyên bố chắc chắn tuyệt đối từ 16 mẫu. Wilson cho
`[80.6, 100]`, tức "100 điểm" chỉ chứng minh được prompt đúng **ít nhất ~81%**.

**Giới hạn đã biết của Wilson:** bao phủ của nó *dao động* và tụt xuống ~92% khi
accuracy sát 0% hoặc 100% (Brown, Cai & DasGupta 2001) — đã kiểm chứng bằng mô
phỏng rằng `statsmodels` tụt y hệt, nên đó là đặc tính của phương pháp chứ không
phải lỗi cài đặt. Khi cần bảo đảm chắc chắn ở vùng biên, dùng
`clopper_pearson_interval()` — phương pháp *exact*, bao phủ ≥ 95% ở **mọi** p,
đổi lại khoảng rộng hơn chút:

```python
wilson_interval(200, 200)           # (98.1, 100.0)
clopper_pearson_interval(200, 200)  # (98.2, 100.0)  <- exact, bảo đảm >= 95%
```

Với kết quả thật của báo cáo (200/200) hai phương pháp chỉ lệch **0.1 điểm**, nên
con số công bố an toàn.

### 3. So hai prompt: McNemar exact (ghép cặp)

```
b = số ca A đúng / B sai        c = số ca A sai / B đúng
n = b + c        k = min(b, c)
p = 2 × Σ(i=0..k) C(n,i) / 2ⁿ
```

Chỉ đếm **ca bất đồng** — ca cả hai cùng đúng (hoặc cùng sai) không mang thông
tin phân biệt. Dùng bản **exact** (nhị thức) chứ không xấp xỉ chi-bình-phương, vì
số ca bất đồng thường rất nhỏ (< 25), chỗ mà xấp xỉ sai. Ca `None` (LLM lỗi) bị
loại cả cặp — nếu giữ, lỗi mạng sẽ bị tính thành bằng chứng hai prompt khác nhau.

### 4. Rút gọn prompt: non-inferiority

```
chấp nhận  ⟺  CI_dưới(prompt_mới) ≥ accuracy(prompt_gốc) − margin
```

**Không được suy `p > 0.05` ⇒ "hai prompt bằng nhau".** Không bác bỏ được H₀ không
có nghĩa H₀ đúng — với bộ mẫu nhỏ thì p *luôn* > 0.05, nên lập luận đó sẽ luôn
kết luận "không đổi" kể cả khi prompt mới tệ đi thật. Cách đúng là **đảo gánh
nặng chứng minh**: chỉ chấp nhận khi cận dưới của khoảng tin cậy vẫn nằm trên
ngưỡng.

### 5. Vừa đúng vừa ngắn

```
vượt   = max(0, (số_từ − word_budget) / word_budget)
điểm   = accuracy − brevity_weight × vượt
```

Chỉ phạt khi **dài hơn** ngân sách; ngắn hơn **không** được thưởng — nếu thưởng,
optimizer sẽ cắt prompt tới mức cụt lủn để ăn điểm. Đếm **từ**, không dùng
tokenizer của model nào: mỗi model tokenize một kiểu nên số token của model A
không so được với model B.

### 6. Đa model

```
điểm       = min( accuracy của từng model )        ← KHÔNG phải trung bình
chênh_lệch = max − min
mẫu số CI  = số MẪU, không phải số_model × số_mẫu
```

Lấy min vì trung bình cho phép model giỏi che model dở: 100 trên model A và 60
trên model B có trung bình 80 — nghe ổn nhưng đó không phải prompt dùng chung
được. Mẫu số không cộng dồn model vì đó là **cùng một bộ mẫu đo lặp**, không phải
quan sát độc lập — cộng dồn sẽ làm khoảng tin cậy hẹp đi giả tạo.

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
benchmark sẽ đo **giọng điệu** thay vì đo quy định. `tests/test_dataset.py` canh
giữ toàn bộ các tính chất này.

Cỡ tập test = 200 không tuỳ tiện — đó là cỡ nhỏ nhất đủ cho mục tiêu "rút ngắn
prompt mà accuracy không tụt quá 5 điểm":

| Cỡ tập test | Khoảng tin cậy (ở 90%) | Kết luận non-inferiority 5đ? |
|---|---|---|
| 16 | ±16.3 | chưa đủ |
| 48 | ±8.8 | chưa đủ |
| 120 | ±5.4 | chưa đủ (chỉ tới 7đ) |
| **200** | **±4.2** | **được** |

Ba file, mở ra xem trực tiếp được:

| File | Nội dung |
|---|---|
| [`examples/tickets.csv`](examples/tickets.csv) | cả 480 ca, chưa chia |
| [`examples/tickets_train.csv`](examples/tickets_train.csv) | 280 ca — optimizer được xem |
| [`examples/tickets_test.csv`](examples/tickets_test.csv) | **200 ca — giữ riêng, optimizer KHÔNG bao giờ thấy** |

Code lúc chạy vẫn gọi `split_samples` chứ không đọc hai file đã chia; chúng chỉ để
người đọc kiểm tra. `tests/test_dataset.py` canh cho chúng luôn khớp chính xác thứ
`split_samples` sinh ra — nếu không, khi bộ mẫu đổi mà quên sinh lại, chúng sẽ âm
thầm thành **dữ liệu ma**: mở ra đọc được, trông đúng, nhưng không phải tập test
thật đã tạo ra con số trong báo cáo.

```python
dev, test = split_samples(samples, test_size=200, seed=0)   # 280 / 200
```

Sinh lại cả ba file: `python -m prompt_tuning_framework.examples.make_tickets`

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
├── tests/              188 test, chạy offline
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
