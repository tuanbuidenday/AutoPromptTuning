# Đo lường: làm sao biết prompt có thật sự tốt hơn

Đây là phần khó nhất của việc tối ưu prompt, và cũng là phần người ta hay làm ẩu
nhất. Sửa prompt rồi thấy điểm cao hơn thì ai cũng tin là mình giỏi lên. Nhưng
"thấy cao hơn" và "thật sự tốt hơn" là hai chuyện khác nhau.

Trước hết, vài từ sẽ gặp suốt tài liệu này:

| Từ | Nghĩa |
|---|---|
| **Ca** | Một dòng dữ liệu, gồm một đoạn văn bản và đáp án đúng của nó |
| **Ca lật** | Ca mà prompt cũ trả lời sai còn prompt mới trả lời đúng (hoặc ngược lại) |
| **Điểm trần** | Điểm kịch khung, tức 100/100 — không thể cao hơn được nữa |
| **Luật lười** | Một mẹo đoán tắt, kiểu "thấy chữ URGENT thì trả lời Yes" |
| **Tập train** | Phần dữ liệu cho optimizer xem để nó sửa prompt |
| **Tập test** | Phần dữ liệu giữ riêng, optimizer không bao giờ được nhìn |

Toàn bộ công thức nằm ở [`core/stats.py`](core/stats.py). Nó chỉ dùng `math` và
`itertools` của thư viện chuẩn Python, không cần scipy hay numpy.

---

## Đo lường không phải một con số

Accuracy trả lời được đúng một câu: "prompt đúng bao nhiêu phần trăm". Nhưng ba
câu quan trọng hơn thì nó chịu:

- Con số này tin được tới đâu?
- Prompt mới hơn prompt cũ, hay chỉ là may?
- Rút prompt cho ngắn lại thì có làm hỏng độ chính xác không?

Mỗi câu cần một công thức khác nhau. Đó là lý do có file này.

---

## 1. Điểm số, và cái bẫy nằm ở mẫu số

```
điểm      = số_ca_đúng / số_ca_CHẤM_ĐƯỢC × 100
đáng tin  = số_ca_chấm_được ≥ min_scored_ratio × tổng_số_ca      (mặc định 0.8)
```

Để ý mẫu số là số ca *chấm được*, không phải tổng số ca. Ca nào gọi LLM bị lỗi
(mạng hỏng, quá hạn ngạch) thì không có kết quả, nên nó rơi khỏi mẫu số.

Nghe thì vô hại, nhưng nó tạo ra thế này: bạn có 12 ca, 9 ca lỗi mạng, 3 ca còn
lại tình cờ đúng hết. Vậy là 3/3 = **100/100**, dù prompt rất dở. Con số đẹp,
hoàn toàn giả, và không có gì báo cho bạn biết.

Cờ `reliable` sinh ra để chặn đúng chỗ đó. Chấm được dưới 80% số ca thì điểm bị
đánh dấu là không đáng tin, và `PromptTuner` từ chối ghi nhận rồi dừng lại, thay
vì tuyên bố thắng lợi trên đống rác.

> Đây không phải lỗi tưởng tượng. AutoPrompt gốc dính đúng bug này:
> `eval/evaluator.py` lọc bỏ các ca `'Discarded'` **trước khi** tính trung bình.

---

## 2. Wilson: một con số không nói lên độ tin cậy

Hãy tưởng tượng bạn nếm cam trong vườn. Nếm 16 quả, cả 16 đều ngọt. Bạn dám
tuyên bố "cả vườn đều ngọt" không? Không — 16 quả là quá ít, có thể bạn nếm trúng
góc ngọt. Bạn chỉ dám nói "ít nhất khoảng 81% số cam là ngọt".

Nếm 200 quả đều ngọt thì mới dám nói "ít nhất 98%".

Đó chính xác là điều Wilson làm: nó biến **một con số** thành **một khoảng**.

```
tâm        = (x + z²/2) / (n + z²)
nửa_rộng   = z/(n + z²) × √( x(n−x)/n + z²/4 )
CI         = [tâm − nửa_rộng,  tâm + nửa_rộng]        x = số đúng, n = số chấm được
```

Cùng là điểm trần 100/100, nhưng ý nghĩa khác hẳn nhau tuỳ số mẫu:

| Đúng | Điểm | Thực ra chỉ chứng minh được |
|---|---|---|
| 16/16 | 100 | ≥ 80.6% |
| 60/60 | 100 | ≥ 94.0% |
| 200/200 | 100 | ≥ 98.1% |

Nói cách khác, "100 điểm" tự nó vô nghĩa nếu không kèm số mẫu.

### Vì sao Wilson chứ không phải công thức trong sách

Công thức phổ biến trong sách giáo khoa tên là Wald: `p ± z√(p(1−p)/n)`. Nó vỡ ở
rìa. Với 16/16 ca đúng thì `p = 1`, nên `√(1 × 0 / 16) = 0`, và khoảng thu về là
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

---

## 3. McNemar: hai prompt có thật sự khác nhau không

Tưởng tượng hai học sinh cùng làm một đề 200 câu. Muốn biết ai giỏi hơn:

- Câu nào cả hai cùng đúng thì không phân biệt được gì
- Câu nào cả hai cùng sai thì cũng vậy
- Chỉ câu nào một người đúng một người sai mới nói lên điều gì

Những câu đó gọi là **ca bất đồng**. McNemar chỉ nhìn vào chúng và bỏ qua phần
còn lại.

```
b = số ca A đúng / B sai        c = số ca A sai / B đúng
n = b + c        k = min(b, c)
p = 2 × Σ(i=0..k) C(n,i) / 2ⁿ
```

### Nó chính là tung đồng xu

Nếu hai prompt thật sự ngang nhau, thì mỗi ca bất đồng giống hệt việc tung một
đồng xu: 50/50 nghiêng về bên nào.

Kết quả thật của framework: **57 ca bất đồng, và cả 57 đều nghiêng về prompt
mới**. Không ca nào xấu đi. Xác suất tung 57 đồng xu ra cùng một mặt là:

```
2 / 2⁵⁷ = 0,0000000000000000139
```

Đó chính là p-value. Không phải ví von — đối chiếu bằng code:

```python
mcnemar_exact(0, 57)   # 1.3877787807814457e-17
2 / 2**57              # 1.3877787807814457e-17   <- khớp từng chữ số
```

Cách đọc: *"Nếu prompt mới không hề tốt hơn, thì để thấy kết quả này bạn phải tung
57 đồng xu ra cùng một mặt."* Nên ta kết luận nó thật sự tốt hơn.

### Vì sao cần ít nhất 6 ca lật

| Số ca lật | Xác suất xảy ra do may | Đủ chưa |
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
là 11 ca đúng thành 16 ca đúng, tức chỉ **5 ca lật**, cho p = 0.0625. Chưa đủ.

### Vì sao không so hai khoảng tin cậy cho nhanh

Vì hai prompt chạy trên **cùng một bộ mẫu**, không phải hai nhóm khác nhau. So
hai khoảng rời nhau là coi chúng như hai thí nghiệm độc lập, tức vứt bỏ thông tin
ghép cặp và trở nên kém nhạy hẳn.

Vài chi tiết cài đặt: dùng bản exact (nhị thức) chứ không xấp xỉ chi-bình-phương,
vì số ca bất đồng thường dưới 25, đúng chỗ mà xấp xỉ sai. Ca lỗi LLM (`None`) bị
loại cả cặp; giữ lại thì lỗi mạng sẽ bị tính thành bằng chứng hai prompt khác
nhau.

---

## 4. Non-inferiority: rút ngắn prompt mà không làm hỏng

Đây là câu hỏi đúng khi bạn muốn prompt gọn hơn: không phải "nó có tốt lên không"
mà là "nó có **tệ đi** không".

```
chấp nhận  ⟺  CI_dưới(prompt_mới) ≥ accuracy(prompt_gốc) − margin
```

Cạm bẫy lớn nhất ở đây: **không được suy `p > 0.05` là "hai prompt bằng nhau"**.
Không bác bỏ được giả thuyết không có nghĩa giả thuyết đó đúng. Với bộ mẫu nhỏ
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

---

## 5. Vừa đúng vừa ngắn

```
vượt   = max(0, (số_từ − word_budget) / word_budget)
điểm   = accuracy − brevity_weight × vượt
```

```python
from prompt_tuning_framework.components import CompositeEvaluator
evaluator = CompositeEvaluator(word_budget=50, brevity_weight=10)
```

Chỉ phạt khi prompt dài hơn ngân sách, còn ngắn hơn thì không được thưởng. Nếu
thưởng, optimizer sẽ cắt prompt tới mức cụt lủn để ăn điểm.

Đếm **từ** chứ không dùng tokenizer của model nào, vì mỗi model tokenize một kiểu
nên số token của model A không so được với model B.

---

## 6. Đa model: lấy min, không lấy trung bình

```
điểm       = min( accuracy của từng model )        ← KHÔNG phải trung bình
chênh_lệch = max − min
mẫu số CI  = số MẪU, không phải số_model × số_mẫu
```

```python
from prompt_tuning_framework.components import MultiModelExecutor, CrossModelEvaluator

executor = MultiModelExecutor(models=[
    {"provider": "google", "model": "gemini-3.1-flash-lite"},
    {"provider": "openai", "model": "gpt-4o-mini"},
], labels=LABELS)
evaluator = CrossModelEvaluator()      # điểm = accuracy của model YẾU NHẤT
```

Lấy min vì trung bình cho phép model giỏi che model dở. Prompt đạt 100 trên model
A và 60 trên model B có trung bình 80, nghe thì ổn, nhưng đó không phải prompt
dùng chung được. Lấy min thì buộc phải sửa cho model yếu nhất.

Mẫu số của khoảng tin cậy không cộng dồn model vào, vì đó là **cùng một bộ mẫu đo
lặp lại** chứ không phải quan sát độc lập. Cộng dồn sẽ làm khoảng hẹp đi một cách
giả tạo.

`metrics["accuracy_spread"]` cho biết prompt kén model tới mức nào.

---

## 7. Tách tập test — nếu không thì mọi con số đều vô nghĩa

Optimizer được xem các ca **sai** để viết lại prompt. Nếu rồi lại chấm điểm trên
chính những ca đó, thì prompt chỉ đang vá thuộc lòng, và 100/100 thu được là điểm
học thuộc, không nói lên gì về ca mới.

```python
from prompt_tuning_framework import split_samples

dev, test = split_samples(samples, test_ratio=0.5, seed=0)  # phân tầng theo nhãn
best = tuner.run(prompt, dev, test_samples=test)            # optimizer không thấy test

print(best.metadata["test_score"])                          # con số đáng công bố
print(best.metadata["test_ci_low"], best.metadata["test_ci_high"])
```

Điểm dev cao hơn điểm test quá 10 điểm thì framework tự log cảnh báo `HỌC THUỘC`.

Đây không phải lo xa. Một lần chạy trên bộ nhỏ cho train 83.3 nhưng test chỉ
66.7. Nếu không tách tập, con số báo cáo đã cao hơn sự thật 16.6 điểm.

---

## 8. Bộ mẫu phải khiến mọi luật lười thất bại

Một benchmark chỉ có giá trị khi không thể ăn gian. Nếu model đoán tắt mà vẫn
được điểm cao thì bạn đang đo mẹo đoán, không đo hiểu biết.

### Bài ticket (tiếng Anh)

[`examples/tickets.csv`](examples/tickets.csv): 480 ca, cân bằng 240 Yes / 240 No,
chia 280 train / 200 test. Sinh bởi `examples/make_tickets.py` với seed cố định
nên tái tạo được.

Quy định ẩn: **Yes khi khách đang trả tiền VÀ bị chặn hoàn toàn.** Bộ mẫu thiết
kế theo kiểu giai thừa, tức là ghép đủ mọi tổ hợp của ba yếu tố *giọng điệu × trả
tiền × bị chặn*, nên không dấu hiệu đơn lẻ nào đủ:

| Luật lười | Điểm đạt được |
|---|---|
| "gào URGENT!!! thì Yes" (giọng điệu) | 50.0 — vô dụng, bằng tung đồng xu |
| "bình tĩnh thì Yes" (giọng điệu) | 50.0 — vô dụng |
| "khách trả tiền thì Yes" | 83.3 — chưa đủ |
| "bị chặn thì Yes" | 83.3 — chưa đủ |
| **"trả tiền VÀ bị chặn thì Yes"** | **100** — luật thật |

Ticket gào to rải đều cả hai nhãn, `P(Yes | gào to) = 50.0%`. Chỗ này quan trọng:
nếu chỉ ticket No mới gào to thì model chỉ cần học "gào to thì No" là xong, và
benchmark đang đo **giọng điệu** thay vì đo quy định. Bộ mẫu đầu tiên hỏng đúng
như vậy — 15/15 ticket gào to đều là No.

### Bài PII (tiếng Việt)

[`examples/pii.csv`](examples/pii.csv): 120 ca, 60 Yes / 60 No, chia 60/60.

Quy định ẩn: **Yes khi văn bản chứa định danh cá nhân của khách** (số điện thoại,
email cá nhân, CCCD, số thẻ, địa chỉ nhà).

Prompt khởi đầu của bài này có sẵn chữ "nhạy cảm", nên bộ mẫu bắt buộc phải rải
chữ đó đều 50/50 giữa hai nhãn. Không làm vậy thì model chỉ cần học "thấy chữ
nhạy cảm thì Yes", và ta đang đo từ khoá chứ không đo việc có lộ dữ liệu:

| Luật lười | Điểm |
|---|---|
| "có chữ nhạy cảm thì Yes" | 50.0 |
| "có chữ khách thì Yes" | 50.0 |
| "có chuỗi số thì Yes" | 54.2 |
| "có dấu @ thì Yes" | 54.2 |

Bốn ô No đều là bẫy thật: bàn *về* chủ đề bảo mật mà không nêu gì cụ thể; mã đơn
hàng dài (có số nhưng không phải định danh); hotline/email công ty; và tên khách
kèm nội dung thường.

`tests/test_dataset.py` và `tests/test_pii_dataset.py` canh giữ toàn bộ các tính
chất này. Chúng đã bắt được lỗi thật: bản đầu của generator PII sinh 120 ca nhưng
chỉ 111 ca duy nhất, khiến 6 văn bản nằm ở cả train lẫn test.

### Vì sao tập test là 200 ca

Không phải con số tuỳ tiện. Đó là cỡ nhỏ nhất đủ để kết luận "rút ngắn prompt mà
accuracy không tụt quá 5 điểm":

| Cỡ tập test | Khoảng tin cậy (ở 90%) | Kết luận non-inferiority 5đ? |
|---|---|---|
| 16 | ±16.3 | chưa đủ |
| 48 | ±8.8 | chưa đủ |
| 120 | ±5.4 | chưa đủ (chỉ tới 7đ) |
| **200** | **±4.2** | **được** |

### Ba file mở ra xem được

| File | Nội dung |
|---|---|
| [`examples/tickets.csv`](examples/tickets.csv) | cả 480 ca, chưa chia |
| [`examples/tickets_train.csv`](examples/tickets_train.csv) | 280 ca — optimizer được xem |
| [`examples/tickets_test.csv`](examples/tickets_test.csv) | 200 ca — giữ riêng, optimizer không bao giờ thấy |

Code lúc chạy vẫn gọi `split_samples` chứ không đọc hai file đã chia; chúng chỉ để
người đọc kiểm tra. `tests/test_dataset.py` canh cho chúng luôn khớp chính xác thứ
`split_samples` sinh ra. Nếu không, khi bộ mẫu đổi mà quên sinh lại, chúng sẽ âm
thầm thành **dữ liệu ma**: mở ra đọc được, trông đúng, nhưng không phải tập test
thật đã tạo ra con số trong báo cáo.

Sinh lại: `python -m prompt_tuning_framework.examples.make_tickets`

---

## 9. Làm sao chắc chắn chính các phép đo này đúng

Đừng tin. Phần thống kê do chính tác giả framework tự viết, mà tự kiểm chứng công
thức bằng chính công thức đó thì vô giá trị — hiểu sai từ đầu thì cả hai lần đều
sai giống hệt nhau.

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
