# 🧩 Prompt Tuning Framework

Framework tối ưu và tinh chỉnh prompt, tự động hoặc bán tự động.

Bạn viết một prompt, đưa cho nó dataset có đáp án, rồi nó tự chạy vòng lặp: chấm
điểm, nhặt ra các câu trả lời sai, nhờ một model mạnh viết lại prompt, rồi lặp
lại. Cuối cùng bạn nhận prompt tốt nhất, kèm bằng chứng thống kê rằng nó thật sự
tốt hơn chứ không phải bạn tưởng thế.

Nó là framework chứ không phải công cụ, vì vòng lặp thuộc về nó chứ không thuộc
về bạn. Bạn chỉ cắm component vào bốn điểm mở rộng và nó gọi ngược lại.

## Mục lục

1. [Cài đặt](#1-cài-đặt) — cài trong một lệnh, chưa cần API key
2. [Cách dùng](#2-cách-dùng) — gắn API key, chọn model, chạy trên terminal và trong Python
3. [Kết quả thật](#3-kết-quả-thật) — 3 bài chạy trên Gemini, kèm khoảng tin cậy và kiểm định
4. [Đo lường](#4-đo-lường-làm-sao-biết-prompt-có-thật-sự-tốt-hơn) — làm sao biết prompt thật sự tốt hơn
5. [Mở rộng](#5-mở-rộng-framework) — cắm component của bạn, cấu trúc bên trong, chạy test
6. [Hỏi đáp](#6-hỏi-đáp) — mỗi câu kèm một lệnh chạy được ngay để tự kiểm chứng

Chỉ muốn dùng thì đọc phần 1 và 2 là đủ.

## Thuật ngữ

Mấy chữ này sẽ gặp suốt tài liệu:

| Thuật ngữ | Nghĩa |
|---|---|
| **sample** | Một dòng dữ liệu: một đoạn văn bản kèm đáp án đúng của nó |
| **flip** | Sample mà prompt cũ trả lời sai còn prompt mới trả lời đúng (hoặc ngược lại) |
| **ceiling score** | Điểm kịch khung, tức 100/100 — không thể cao hơn được nữa |
| **shortcut rule** | Một mẹo đoán tắt, kiểu "thấy chữ URGENT thì trả lời Yes" |
| **train set** | Phần dữ liệu cho optimizer xem để nó sửa prompt |
| **test set** | Phần dữ liệu giữ riêng, optimizer không bao giờ được nhìn |
| **discordant pair** | Sample mà hai prompt trả lời khác nhau — chỉ những sample này mới mang thông tin phân biệt |
| **coverage** | Tỉ lệ khoảng tin cậy "trúng" giá trị thật; khoảng 95% thì phải trúng ~95% số lần |

---

# 1. Cài đặt

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

## Extras (tuỳ chọn)

| Extras | Thêm gì | Khi nào cần |
|---|---|---|
| `[test]` | pytest | Chạy bộ test |
| `[verify]` | statsmodels, scipy, numpy | Đối chiếu phần thống kê với thư viện độc lập |
| `[all]` | cả hai cái trên | Bạn muốn đủ đồ nghề để phát triển |
| `[autoprompt]` | easydict, langchain <0.3 | Chỉ khi bạn dùng plugin optimizer AutoPrompt |

Tôi cố ý để `[autoprompt]` đứng ngoài base install và ngoài `[all]`, vì nó ghim
`langchain<0.3` và sẽ kéo tụt SDK Gemini về đời 2024 cho cả người không dùng tới
nó. Lý do đầy đủ ở [phần 5](#quan-hệ-với-autoprompt).

> Nếu dùng cú pháp `pip install "tên_gói[extras] @ git+..."` thì extras phải đứng
> trước dấu `@` theo chuẩn PEP 508. Viết thành
> `...#subdirectory=prompt_tuning_framework[test]` thì pip coi `[test]` là một
> phần của tên thư mục và lặng lẽ bỏ qua extras — cài xong mà thiếu, không báo gì.

---

# 2. Cách dùng

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

Vì sao chuyện này quan trọng: xem [phần 4](#4-đo-lường-làm-sao-biết-prompt-có-thật-sự-tốt-hơn).

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

Kết quả của các bài này: xem [phần 3](#3-kết-quả-thật).

---

# 3. Kết quả thật

Mọi lần chạy dưới đây đều dùng Gemini thật. Điểm công bố lấy từ **test set** mà
optimizer chưa từng nhìn thấy.

Cách đọc các con số này (khoảng tin cậy là gì, McNemar là gì, vì sao lấy min chứ
không lấy trung bình): xem [phần 4](#4-đo-lường-làm-sao-biết-prompt-có-thật-sự-tốt-hơn).

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

## Tổng hợp

| | |
|---|---|
| Test | 205 passed, 3 skipped |
| Bài đã đo bằng LLM thật | 2 miền — ticket tiếng Anh, PII tiếng Việt |
| Optimizer đã kiểm chứng | 2 — `llm_rewrite`, plugin `autoprompt` |
| Chi phí một lần chạy đầy đủ | ~1.300 VND |

---

# 4. Đo lường: làm sao biết prompt có thật sự tốt hơn

Đây là phần khó nhất của việc tối ưu prompt, và cũng là phần người ta hay làm ẩu
nhất. Sửa prompt rồi thấy điểm cao hơn thì ai cũng tin là mình giỏi lên. Nhưng
"thấy cao hơn" và "thật sự tốt hơn" là hai chuyện khác nhau.

Toàn bộ công thức nằm ở [`core/stats.py`](core/stats.py). Nó chỉ dùng `math` và
`itertools` của thư viện chuẩn Python, không cần scipy hay numpy.

Accuracy trả lời được đúng một câu: "prompt đúng bao nhiêu phần trăm". Nhưng ba
câu quan trọng hơn thì nó chịu:

- Con số này tin được tới đâu?
- Prompt mới hơn prompt cũ, hay chỉ là may?
- Rút prompt cho ngắn lại thì có làm hỏng độ chính xác không?

Mỗi câu cần một công thức khác nhau. Đó là lý do có phần này.

## 4.1 Điểm số, và cái bẫy nằm ở mẫu số

```
score     = num_correct / num_SCORED * 100
reliable  = num_scored >= min_scored_ratio * num_samples      (mặc định 0.8)
```

Để ý mẫu số là `num_scored`, tức số sample *chấm được*, không phải tổng số
sample. Sample nào gọi LLM bị lỗi (mạng hỏng, quá hạn ngạch) thì không có kết
quả, nên nó rơi khỏi mẫu số.

Nghe thì vô hại, nhưng nó tạo ra thế này: bạn có 12 sample, 9 sample lỗi mạng, 3
sample còn lại tình cờ đúng hết. Vậy là 3/3 = **100/100**, dù prompt rất dở. Con
số đẹp, hoàn toàn giả, và không có gì báo cho bạn biết.

Cờ `reliable` sinh ra để chặn đúng chỗ đó. Chấm được dưới 80% số sample thì điểm
bị đánh dấu là không đáng tin, và `PromptTuner` từ chối ghi nhận rồi dừng lại,
thay vì tuyên bố thắng lợi trên đống rác.

> Đây không phải lỗi tưởng tượng. AutoPrompt gốc dính đúng bug này:
> `eval/evaluator.py` lọc bỏ các sample `'Discarded'` **trước khi** tính trung bình.

## 4.2 Wilson: một con số không nói lên độ tin cậy

Hãy tưởng tượng bạn nếm cam trong vườn. Nếm 16 quả, cả 16 đều ngọt. Mười sáu quả
vẫn là quá ít để tuyên bố "cả vườn đều ngọt" — biết đâu bạn nếm trúng góc ngọt.
Bạn chỉ dám nói "ít nhất khoảng 81% số cam là ngọt".

Nếm 200 quả đều ngọt thì mới dám nói "ít nhất 98%".

Đó chính xác là điều Wilson làm: nó biến **một con số** thành **một khoảng**.

```
center     = (x + z²/2) / (n + z²)
half_width = z/(n + z²) * sqrt( x(n−x)/n + z²/4 )
CI         = [center − half_width, center + half_width]

x = num_correct, n = num_scored, z = 1.96 cho mức 95%
```

Cùng là ceiling score 100/100, nhưng ý nghĩa khác hẳn nhau tuỳ số sample:

| Đúng | Điểm | Thực ra chỉ chứng minh được |
|---|---|---|
| 16/16 | 100 | ≥ 80.6% |
| 60/60 | 100 | ≥ 94.0% |
| 200/200 | 100 | ≥ 98.1% |

Nói cách khác, "100 điểm" tự nó vô nghĩa nếu không kèm số sample.

### Vì sao Wilson chứ không phải công thức trong sách

Công thức phổ biến trong sách giáo khoa tên là Wald: `p ± z√(p(1−p)/n)`. Nó vỡ ở
rìa. Với 16/16 sample đúng thì `p = 1`, nên `√(1 × 0 / 16) = 0`, và khoảng thu về là
`[100, 100]` — tức "chắc chắn tuyệt đối, không bao giờ sai", suy ra từ vỏn vẹn 16
mẫu. Vô lý. Wilson sinh ra để sửa đúng chỗ đó.

### Giới hạn đã biết của Wilson

Khoảng tin cậy 95% lẽ ra phải "trúng" khoảng 95% số lần. Người ta gọi đó là **độ
bao phủ**. Bao phủ của Wilson dao động, và tụt xuống khoảng 92% khi accuracy sát
0% hoặc 100% (Brown, Cai & DasGupta 2001).

Đây là đặc tính của phương pháp, không phải lỗi cài đặt — có mô phỏng chứng minh
`statsmodels` tụt y hệt. Khi cần bảo đảm chắc chắn ở vùng biên thì dùng
`clopper_pearson_interval()`, một phương pháp *exact*, bao phủ luôn ≥ 95% ở mọi
mức, đổi lại khoảng rộng hơn một chút:

```python
wilson_interval(200, 200)           # (98.1, 100.0)
clopper_pearson_interval(200, 200)  # (98.2, 100.0)  <- exact, bảo đảm >= 95%
```

Với kết quả thật của báo cáo (200/200), hai phương pháp chỉ lệch 0.1 điểm, nên
con số công bố an toàn.

## 4.3 McNemar: hai prompt có thật sự khác nhau không

Tưởng tượng hai học sinh cùng làm một đề 200 câu. Muốn biết ai giỏi hơn:

- Câu nào cả hai cùng đúng thì không phân biệt được gì
- Câu nào cả hai cùng sai thì cũng vậy
- Chỉ câu nào một người đúng một người sai mới nói lên điều gì

Những câu đó gọi là **discordant pair**. McNemar chỉ nhìn vào chúng và bỏ qua phần
còn lại.

```
b = num_samples(A correct, B wrong)     c = num_samples(A wrong, B correct)
n = b + c                               k = min(b, c)
p = 2 * Σ(i=0..k) C(n,i) / 2ⁿ
```

### Nó chính là tung đồng xu

Nếu hai prompt thật sự ngang nhau, thì mỗi discordant pair giống hệt việc tung một
đồng xu: 50/50 nghiêng về bên nào.

Kết quả thật của framework: **57 discordant pair, và cả 57 đều nghiêng về prompt
mới**. Không sample nào xấu đi. Xác suất tung 57 đồng xu ra cùng một mặt là:

```
2 / 2⁵⁷ = 0,0000000000000000139
```

Đó chính là p-value. Không phải ví von — đối chiếu bằng code:

```python
mcnemar_exact(0, 57)   # 1.3877787807814457e-17
2 / 2**57              # 1.3877787807814457e-17   <- khớp từng chữ số
```

Cách đọc: *"Nếu prompt mới không hề tốt hơn, thì để thấy kết quả này bạn phải tung
57 đồng xu ra cùng một mặt."* Nên tôi kết luận nó thật sự tốt hơn.

### Vì sao cần ít nhất 6 flip

| Số flip | Xác suất xảy ra do may | Đủ chưa |
|---|---|---|
| 5 | 2/2⁵ = 6,25% | chưa — quá dễ xảy ra |
| 6 | 2/2⁶ = 3,13% | được, dưới ngưỡng 5% |
| 20 | 0,0002% | được (bài PII) |
| 57 | gần như bằng 0 | được (bài ticket) |

```python
from prompt_tuning_framework import min_flips_for_significance
min_flips_for_significance(0.05)   # 6
```

Đây cũng là lý do con số cũ trong slide không dùng được: 16 mẫu, 68.8 → 100 nghĩa
là 11 sample đúng thành 16 sample đúng, tức chỉ **5 flip**, cho p = 0.0625. Chưa đủ.

### Vì sao không so hai khoảng tin cậy cho nhanh

Vì hai prompt chạy trên **cùng một dataset**, không phải hai nhóm khác nhau. So
hai khoảng rời nhau là coi chúng như hai thí nghiệm độc lập, tức vứt bỏ thông tin
ghép cặp và trở nên kém nhạy hẳn.

Vài chi tiết cài đặt: dùng bản exact (nhị thức) chứ không xấp xỉ chi-bình-phương,
vì số discordant pair thường dưới 25, đúng chỗ mà xấp xỉ sai. Sample lỗi LLM (`None`) bị
loại cả cặp; giữ lại thì lỗi mạng sẽ bị tính thành bằng chứng hai prompt khác
nhau.

## 4.4 Non-inferiority: rút ngắn prompt mà không làm hỏng

Đây là câu hỏi đúng khi bạn muốn prompt gọn hơn: không phải "nó có tốt lên không"
mà là "nó có **tệ đi** không".

```
accept  ⟺  CI_lower(new_prompt) >= accuracy(base_prompt) − margin
```

Cạm bẫy lớn nhất ở đây: **không được suy `p > 0.05` là "hai prompt bằng nhau"**.
Không bác bỏ được giả thuyết không có nghĩa giả thuyết đó đúng. Với dataset nhỏ
thì p *luôn* lớn hơn 0.05, nên lập luận đó sẽ **luôn** kết luận "không đổi", kể
cả khi prompt mới tệ đi thật.

Cách đúng là đảo ngược gánh nặng chứng minh: chỉ chấp nhận khi cận dưới của
khoảng tin cậy prompt mới vẫn nằm trên ngưỡng.

```python
from prompt_tuning_framework import non_inferiority

non_inferiority(base_correct=15, new_correct=15, num_total=16, margin_pp=5.0)
# False — điểm y hệt nhau, nhưng 16 mẫu không đủ để kết luận

non_inferiority(base_correct=180, new_correct=180, num_total=200, margin_pp=5.0)
# True
```

## 4.5 Vừa đúng vừa ngắn

```
excess = max(0, (num_words − word_budget) / word_budget)
score  = accuracy − brevity_weight * excess
```

```python
from prompt_tuning_framework.components import CompositeEvaluator
evaluator = CompositeEvaluator(word_budget=50, brevity_weight=10)
```

Chỉ phạt khi prompt dài hơn ngân sách, còn ngắn hơn thì không được thưởng. Nếu
thưởng, optimizer sẽ cắt prompt tới mức cụt lủn để ăn điểm.

Đếm **từ** chứ không dùng tokenizer của model nào, vì mỗi model tokenize một kiểu
nên số token của model A không so được với model B.

## 4.6 Đa model: lấy min, không lấy trung bình

```
score            = min( accuracy của từng model )      <- KHÔNG phải mean
spread           = max − min
CI denominator   = num_samples, KHÔNG phải num_models * num_samples
```

```python
from prompt_tuning_framework.components import MultiModelExecutor, CrossModelEvaluator

executor = MultiModelExecutor(models=[
    {"provider": "google", "model": "gemini-3.1-flash-lite"},
    {"provider": "openai", "model": "gpt-4o-mini"},
], labels=LABELS)
evaluator = CrossModelEvaluator()      # score = accuracy của model YẾU NHẤT
```

Lấy min vì trung bình cho phép model giỏi che model dở. Prompt đạt 100 trên model
A và 60 trên model B có trung bình 80, nghe thì ổn, nhưng đó không phải prompt
dùng chung được. Lấy min thì buộc phải sửa cho model yếu nhất.

Mẫu số của khoảng tin cậy không cộng dồn model vào, vì đó là **cùng một dataset đo
lặp lại** chứ không phải quan sát độc lập. Cộng dồn sẽ làm khoảng hẹp đi một cách
giả tạo.

`metrics["accuracy_spread"]` cho biết prompt kén model tới mức nào.

## 4.7 Tách test set — nếu không thì mọi con số đều vô nghĩa

Optimizer được xem các sample **sai** để viết lại prompt. Nếu rồi lại chấm điểm trên
chính những sample đó, thì prompt chỉ đang vá thuộc lòng, và 100/100 thu được là điểm
học thuộc, không nói lên gì về sample mới.

```python
from prompt_tuning_framework import split_samples

dev, test = split_samples(samples, test_ratio=0.5, seed=0)  # phân tầng theo nhãn
best = tuner.run(prompt, dev, test_samples=test)            # optimizer không thấy test

print(best.metadata["test_score"])                          # con số đáng công bố
print(best.metadata["test_ci_low"], best.metadata["test_ci_high"])
```

Điểm dev cao hơn điểm test quá 10 điểm thì framework tự log cảnh báo `HỌC THUỘC`.

Đây không phải lo xa. Một lần chạy trên dataset nhỏ cho train 83.3 nhưng test chỉ
66.7. Nếu không tách tập, con số báo cáo đã cao hơn sự thật 16.6 điểm.

## 4.8 Dataset phải khiến mọi shortcut rule thất bại

Một benchmark chỉ có giá trị khi không thể ăn gian. Nếu model đoán tắt mà vẫn
được điểm cao thì bạn đang đo mẹo đoán, không đo hiểu biết.

### Bài ticket (tiếng Anh)

[`examples/tickets.csv`](examples/tickets.csv): 480 sample, cân bằng 240 Yes / 240 No,
chia 280 train / 200 test. Sinh bởi `examples/make_tickets.py` với seed cố định
nên tái tạo được.

Quy định ẩn: **Yes khi khách đang trả tiền VÀ bị chặn hoàn toàn.** Dataset thiết
kế theo kiểu giai thừa, tức là ghép đủ mọi tổ hợp của ba yếu tố *giọng điệu × trả
tiền × bị chặn*, nên không dấu hiệu đơn lẻ nào đủ:

| Shortcut rule | Điểm đạt được |
|---|---|
| "gào URGENT!!! thì Yes" (giọng điệu) | 50.0 — vô dụng, bằng tung đồng xu |
| "bình tĩnh thì Yes" (giọng điệu) | 50.0 — vô dụng |
| "khách trả tiền thì Yes" | 83.3 — chưa đủ |
| "bị chặn thì Yes" | 83.3 — chưa đủ |
| **"trả tiền VÀ bị chặn thì Yes"** | **100** — luật thật |

Ticket gào to rải đều cả hai nhãn, `P(Yes | gào to) = 50.0%`. Chỗ này quan trọng:
nếu chỉ ticket No mới gào to thì model chỉ cần học "gào to thì No" là xong, và
benchmark đang đo **giọng điệu** thay vì đo quy định. Dataset đầu tiên hỏng đúng
như vậy — 15/15 ticket gào to đều là No.

### Bài PII (tiếng Việt)

[`examples/pii.csv`](examples/pii.csv): 120 sample, 60 Yes / 60 No, chia 60/60.

Quy định ẩn: **Yes khi văn bản chứa định danh cá nhân của khách** (số điện thoại,
email cá nhân, CCCD, số thẻ, địa chỉ nhà).

Prompt khởi đầu của bài này có sẵn chữ "nhạy cảm", nên dataset bắt buộc phải rải
chữ đó đều 50/50 giữa hai nhãn. Không làm vậy thì model chỉ cần học "thấy chữ
nhạy cảm thì Yes", và tôi đang đo từ khoá chứ không đo việc có lộ dữ liệu:

| Shortcut rule | Điểm |
|---|---|
| "có chữ nhạy cảm thì Yes" | 50.0 |
| "có chữ khách thì Yes" | 50.0 |
| "có chuỗi số thì Yes" | 54.2 |
| "có dấu @ thì Yes" | 54.2 |

Bốn ô No đều là bẫy thật: bàn *về* chủ đề bảo mật mà không nêu gì cụ thể; mã đơn
hàng dài (có số nhưng không phải định danh); hotline/email công ty; và tên khách
kèm nội dung thường.

`tests/test_dataset.py` và `tests/test_pii_dataset.py` canh giữ toàn bộ các tính
chất này. Chúng đã bắt được lỗi thật: bản đầu của generator PII sinh 120 sample nhưng
chỉ 111 sample duy nhất, khiến 6 văn bản nằm ở cả train lẫn test.

### Vì sao test set là 200 sample

Không phải con số tuỳ tiện. Đó là cỡ nhỏ nhất đủ để kết luận "rút ngắn prompt mà
accuracy không tụt quá 5 điểm":

| Cỡ test set | Khoảng tin cậy (ở 90%) | Kết luận non-inferiority 5đ? |
|---|---|---|
| 16 | ±16.3 | chưa đủ |
| 48 | ±8.8 | chưa đủ |
| 120 | ±5.4 | chưa đủ (chỉ tới 7đ) |
| **200** | **±4.2** | **được** |

### Ba file mở ra xem được

| File | Nội dung |
|---|---|
| [`examples/tickets.csv`](examples/tickets.csv) | cả 480 sample, chưa chia |
| [`examples/tickets_train.csv`](examples/tickets_train.csv) | 280 sample — optimizer được xem |
| [`examples/tickets_test.csv`](examples/tickets_test.csv) | 200 sample — giữ riêng, optimizer không bao giờ thấy |

Code lúc chạy vẫn gọi `split_samples` chứ không đọc hai file đã chia; chúng chỉ để
người đọc kiểm tra. `tests/test_dataset.py` canh cho chúng luôn khớp chính xác thứ
`split_samples` sinh ra. Nếu không, khi dataset đổi mà quên sinh lại, chúng sẽ âm
thầm thành **dữ liệu ma**: mở ra đọc được, trông đúng, nhưng không phải test set
thật đã tạo ra con số trong báo cáo.

Sinh lại: `python -m prompt_tuning_framework.examples.make_tickets`

## 4.9 Làm sao chắc chắn chính các phép đo này đúng

Đừng tin. Phần thống kê do chính tôi viết, mà tự kiểm chứng công thức bằng chính
công thức đó thì vô giá trị — hiểu sai từ đầu thì cả hai lần đều sai giống hệt
nhau.

[`tests/test_metrics_verification.py`](tests/test_metrics_verification.py) kiểm
chứng bằng ba tầng độc lập, và CI chạy lại mỗi lần push:

| Tầng | Cách làm | Bắt được lỗi gì |
|---|---|---|
| 1. Đối chiếu ngoài | So với `statsmodels`, thư viện do người khác viết | Cài sai công thức |
| 2. Giá trị tham chiếu | Khoá cứng giá trị suy tay (`b=0, c=k` thì `p = 2/2ᵏ`) | Cài sai, kể cả khi máy không có statsmodels |
| 3. Mô phỏng | Kiểm tra tính chất, không kiểm tra công thức | Dùng sai công cụ — hai tầng trên bỏ lọt |

Tầng 3 mạnh nhất vì nó không dựa vào công thức nào cả: khoảng tin cậy 95% thì
phải chứa giá trị thật khoảng 95% số lần; kiểm định mức 5% thì chỉ được báo động
sai tối đa 5% số lần. Có cả chiều ngược lại — McNemar phải **bắt được** khác biệt
thật trên 95% số lần, vì nếu không thì một hàm luôn trả về `p = 1.0` cũng qua
được bài kiểm tra báo-động-giả.

Kết quả: Wilson và McNemar khớp `statsmodels` sai lệch 0.000000. Clopper-Pearson
tự viết (dò nhị phân trên CDF nhị thức) khớp bản của statsmodels (phân vị Beta
qua scipy) tới 10⁻¹⁴ — hai đường tính hoàn toàn khác nhau mà ra cùng kết quả.

```bash
pip install -e "prompt_tuning_framework/[test,verify]"
pytest prompt_tuning_framework/tests/test_metrics_verification.py -v
```

`statsmodels` không phải dependency của framework. `core/stats.py` chỉ dùng thư
viện chuẩn. Nó chỉ cần khi muốn *chứng minh* phép đo đúng.

---

# 5. Mở rộng framework

Phần này dành cho bạn nếu bạn muốn cắm component của mình vào, hoặc muốn hiểu bên
trong framework chạy thế nào. Chỉ dùng framework thì không cần đọc.

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

## Dùng bằng YAML

```python
from prompt_tuning_framework import tuner_from_yaml
tuner = tuner_from_yaml("prompt_tuning_framework/examples/config_example.yml")
best = tuner.run(initial_prompt, samples)
```

Đổi `optimizer.name` giữa `autoprompt` và `llm_rewrite` mà không sửa dòng code
nào.

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

---

# 6. Hỏi đáp

Đây là những câu tôi đoán sẽ bị hỏi, kèm câu trả lời và một lệnh để chứng minh
ngay tại chỗ. Bạn không cần tin lời tôi, cứ chạy thử.

Các lệnh chạy từ thư mục gốc repo. Trừ câu cuối, tất cả đều không gọi LLM nên
không tốn tiền và xong trong vài giây.

## Đây là framework hay chỉ là một ứng dụng?

Framework. Khác biệt nằm ở chỗ ai giữ vòng lặp chính.

Với một ứng dụng, bạn là người gọi nó. Với framework thì ngược lại: nó giữ vòng
lặp và gọi ngược vào code của bạn. Người ta gọi cái đó là Inversion of Control.
Ở đây bạn chỉ cắm component vào bốn điểm, còn vòng lặp không thuộc về bạn.

```bash
grep -n "executor.execute\|evaluator.evaluate\|store.record_score\|optimizer.propose" \
  prompt_tuning_framework/core/tuner.py
```

Lệnh in ra khoảng 10 dòng. Bốn dòng đầu (7–10) là docstring mô tả vòng lặp. Bốn
dòng quan trọng là 102, 104, 121, 146 — chúng nằm bên trong vòng `for` của
framework. Hai dòng cuối là lúc chấm test set.

Điểm cần thấy: cả bốn lời gọi đó đều nằm trong code của framework, không phải
code của người dùng. Người dùng chỉ đưa component vào rồi gọi `tuner.run()` đúng
một lần.

## AutoPrompt đứng ở đâu trong kiến trúc?

Nó là một plugin optimizer, ngang hàng với optimizer tôi tự viết. Đổi qua lại
bằng một dòng YAML. Nếu AutoPrompt nằm ở lõi thì cái này đã là ứng dụng bọc quanh
AutoPrompt, chứ không phải framework.

```bash
python -c "
import prompt_tuning_framework.components
from prompt_tuning_framework import available
for k in ('store','executor','evaluator','optimizer'):
    print(f'{k:10}:', available(k))
"
```

Kết quả in ra `optimizer: ['autoprompt', 'llm_rewrite']`. Nó là một phần tử trong
danh sách, không phải trung tâm.

Mà không chỉ đăng ký được — nó chạy thật. Engine AutoPrompt cắm vào qua interface
`BaseOptimizer` đạt 85.0/100 trên test set, CI 95% [79.4, 89.3] (train 60 / test
200). Chi tiết ở [phần 3](#plugin-autoprompt--engine-ngoài-cắm-vào). Lõi
framework không sửa dòng nào.

Nếu bị vặn "sao plugin chỉ 85 mà optimizer nhà được 100?", tôi trả lời thẳng: hai
lần chạy khác điều kiện, một bên 280 sample train một bên 60, nên không so trực
tiếp được. Tôi cũng không dùng nó để nói engine nào hơn. Nó chỉ trả lời đúng một
câu: engine ngoài có cắm vào chạy được không.

## Prompt tối ưu có thật sự tốt hơn, hay chỉ may?

McNemar ghép cặp trên test set cho p = 1.4 × 10⁻¹⁷. Cụ thể là 57 sample chuyển từ sai
sang đúng, không sample nào xấu đi.

Cách hiểu con số đó: nếu hai prompt thực sự ngang nhau thì mỗi discordant pair giống như
tung một đồng xu. Ở đây có 57 discordant pair và cả 57 đều nghiêng về prompt mới. Xác
suất tung 57 đồng xu ra cùng một mặt chính là 2/2⁵⁷, tức đúng bằng p-value.

```bash
python -c "
from prompt_tuning_framework import mcnemar_exact, min_flips_for_significance
print('Ket qua that (57 flips, 0 regressions):', mcnemar_exact(0, 57))
print('Tung 57 dong xu ra cung mot mat       :', 2 / 2**57)
print('Nguong can de dat p<0.05              :', min_flips_for_significance(0.05), 'flips')
"
```

Dùng McNemar chứ không so hai khoảng tin cậy rời nhau, vì hai prompt chạy trên
cùng một dataset. Sample nào cả hai cùng đúng thì không giúp phân biệt được gì.

## Điểm 100/100 có đáng tin không?

Nghĩ thế này cho dễ. Bạn nếm cam trong vườn, nếm 16 quả đều ngọt. Mười sáu quả là
quá ít để dám tuyên bố cả vườn ngọt — bạn chỉ dám nói "ít nhất khoảng 81% là
ngọt". Nếm 200 quả đều ngọt thì mới dám nói "ít nhất 98%".

Điểm số cũng vậy, nên tôi không bao giờ báo nó trần trụi mà luôn kèm khoảng tin
cậy. Trên 200 sample, 100 điểm cho khoảng [98.1, 100.0], tức chỉ chứng minh được
prompt đúng ít nhất 98.1%. Cũng 100 điểm đó mà chỉ có 16 sample thì tụt xuống ≥ 80.6%.

```bash
python -c "
from prompt_tuning_framework import wilson_interval
for n in (16, 60, 200):
    lo, hi = wilson_interval(n, n)
    print(f'{n:>3}/{n:<3} = 100 diem  ->  that ra chi chung minh duoc >= {lo:.1f}%')
"
```

Nói cách khác, "100 điểm" tự nó không có nghĩa gì nếu không kèm số sample.

## Sao biết prompt không phải học thuộc dataset?

Vì có test set giữ riêng, và optimizer không bao giờ nhìn thấy nó. Optimizer chỉ
được xem các sample sai của train set. Điểm công bố thì lấy từ test set. Train 100 mà
test cũng 100, nên không có chuyện học thuộc.

Đây không phải lo xa. Có một lần chạy trên dataset nhỏ cho train 83.3 nhưng test
chỉ 66.7. Nếu không tách tập, tôi đã báo cáo con số cao hơn sự thật 16.6 điểm mà
không hề biết.

```bash
python -m pytest prompt_tuning_framework/tests/test_train_test_split.py -v -k "never_sees or overfitting"
```

Có test khoá chặt chuyện "optimizer không bao giờ thấy test set", và test chứng
minh framework tự cảnh báo HỌC THUỘC khi điểm dev vượt test quá 10 điểm.

## Dataset có dễ quá không, model đoán mò cũng đúng chăng?

Không, vì dataset thiết kế theo kiểu giai thừa để mọi shortcut rule đều thất bại. Quy
định ẩn là khách phải vừa trả tiền vừa bị chặn hoàn toàn. Bám vào một dấu hiệu
đơn lẻ chỉ được 83.3 điểm; muốn 100 thì phải hiểu đúng sự kết hợp.

```bash
python -m pytest prompt_tuning_framework/tests/test_dataset.py -v
```

Đáng chú ý nhất là `test_tone_is_useless_for_guessing`. Ticket gào "URGENT!!!"
được rải đều cả Yes lẫn No, đúng 50/50, nên đoán theo giọng điệu là vô dụng.

Dataset đầu tiên tôi làm hỏng đúng chỗ này: 15/15 ticket gào to đều là No. Model
chỉ cần học "gào to thì No" là xong, và lúc đó benchmark đang đo giọng điệu chứ
không đo quy định. Test này sinh ra để lỗi đó không quay lại.

Dataset PII tiếng Việt cũng vậy, xem `test_pii_dataset.py`. Prompt khởi đầu có
sẵn chữ "nhạy cảm" nên dataset phải rải chữ đó đều 50/50, nếu không thì tôi đang
đo từ khoá chứ không đo việc có lộ dữ liệu hay không.

## Làm sao chắc chắn phép đo của tôi đúng?

Đừng tin tôi. Phần thống kê do tôi tự viết, nên nó được kiểm chứng bằng ba tầng
độc lập, và CI chạy lại mỗi lần push.

```bash
pip install -e "prompt_tuning_framework/[test,verify]"
python -m pytest prompt_tuning_framework/tests/test_metrics_verification.py -v
```

| Tầng | Cách làm | Bắt được lỗi gì |
|---|---|---|
| 1. Đối chiếu | So với `statsmodels`, thư viện do người khác viết | Cài sai công thức |
| 2. Tham chiếu | Khoá cứng giá trị suy tay (`b=0, c=k` thì `p = 2/2ᵏ`) | Cài sai, kể cả khi máy không có statsmodels |
| 3. Mô phỏng | Kiểm tra tính chất thống kê, không kiểm tra công thức | Dùng sai công cụ |

Tầng 3 mạnh nhất vì nó không dựa vào công thức nào cả: khoảng tin cậy 95% thì
phải chứa giá trị thật khoảng 95% số lần, không thì sai. Có cả chiều ngược lại
nữa — McNemar phải bắt được khác biệt thật trên 95% số lần, vì nếu không thì một
hàm luôn trả về `p = 1.0` cũng qua được bài kiểm tra.

Kết quả: Wilson và McNemar khớp statsmodels, lệch 0.000000.

Chi tiết hơn: [phần 4.9](#49-làm-sao-chắc-chắn-chính-các-phép-đo-này-đúng).

## Điểm cao thế, có phải nhờ dùng model xịn không?

Không. Executor dùng model rẻ nhất (`gemini-3.1-flash-lite`), và nó là thứ chạy
1.320 lượt. Model mạnh chỉ dùng cho optimizer, mà optimizer chỉ gọi 1 lần mỗi
vòng, cả run có 3 lần. Nhờ sự bất đối xứng đó mà một lần chạy đầy đủ chỉ tốn
khoảng 1.300 VND.

```bash
grep -n "DEFAULT_MODELS" -A 4 prompt_tuning_framework/llm.py
```

Prompt tối ưu cũng chuyển được sang model khác. Đo trên 3 model, model yếu nhất
vẫn đạt 95.0, và độ chênh giữa các model giảm từ 8.4 xuống 5.0. Tức là nó ít kén
model hơn prompt gốc, dù chỉ được tinh chỉnh trên đúng một model.

## Sao chấm đa model bằng min mà không lấy trung bình?

Vì trung bình cho phép model giỏi che lấp model dở. Một prompt đạt 100 trên model
A và 60 trên model B có trung bình 80, nghe thì ổn, nhưng đó không phải prompt
dùng chung được. Lấy min buộc phải sửa cho model yếu nhất.

```bash
python -m pytest prompt_tuning_framework/tests/test_metrics.py -v -k cross_model
```

Trong đó có một test khoá chi tiết dễ bỏ sót: mẫu số của khoảng tin cậy phải là
số sample, không phải số_model × số_mẫu. Cộng dồn vào sẽ làm khoảng hẹp đi một cách
giả tạo, vì đó là cùng một dataset đo lặp lại chứ không phải quan sát độc lập.

## Cho xem chạy thật được không?

Được. Dataset đầy đủ tốn khoảng 1.320 lượt gọi nên hơi lâu, còn bản rút gọn thì
vài phút là xong:

```bash
python -m prompt_tuning_framework.examples.hard_example --nho 24 --max-iters 2
```

Dataset đầy đủ, cần API key đã bật billing, khoảng 1.300 VND:

```bash
python -m prompt_tuning_framework.examples.hard_example --delay 0 --workers 8
```

Bài tiếng Việt (phát hiện lộ thông tin khách hàng) nhẹ hơn, 120 sample:

```bash
python -m prompt_tuning_framework.examples.pii_example
```

## Số liệu cần nhớ

Bài ticket, 480 sample chia 280 train / 200 test:

| | Train | Test (optimizer chưa từng thấy) |
|---|---|---|
| Prompt gốc | 68.9 | 71.5 — CI [64.9, 77.3] |
| Prompt tối ưu | 100.0 | 100.0 — CI [98.1, 100.0] |

McNemar p = 1.4 × 10⁻¹⁷, 57 flip, 0 sample xấu đi.

Bài PII tiếng Việt, 120 sample chia 60/60: test 66.7 → 100.0, CI [94.0, 100.0],
McNemar p = 1.9 × 10⁻⁶.

Tổng: 205 test, chi phí khoảng 1.300 VND một lần chạy đầy đủ.

## Nếu bị hỏi điểm yếu

Cứ nói thẳng. Người chấm thường đánh giá cao chuyện tôi tự biết giới hạn của mình
hơn là cố che.

Dataset sinh bằng generator chứ không phải ticket thật. Nó kiểm tra được khả năng
suy ra quy định ẩn, nhưng ngôn ngữ đơn giản hơn ticket ngoài đời.

Bao phủ của Wilson tụt xuống khoảng 92% khi accuracy sát 100%. Đây là đặc tính đã
biết của phương pháp chứ không phải bug — tôi chạy cùng mô phỏng đó với
statsmodels và nó tụt y hệt. Riêng kết quả 200/200 vẫn an toàn, vì Wilson cho
[98.1, 100] gần trùng với Clopper-Pearson [98.2, 100].

Mới chỉ thử trên bài phân loại 2 nhãn. Đã đo trên hai miền khác nhau (ticket
tiếng Anh và PII tiếng Việt) nhưng cả hai đều 2 nhãn. Framework không ràng buộc
điều đó, chỉ là tôi chưa đo bài nhiều nhãn hay bài sinh văn bản.

Muốn siết non-inferiority xuống margin 3 điểm thì cần hơn 400 sample test. Dataset
hiện tại chỉ đủ cho margin 5 điểm.
